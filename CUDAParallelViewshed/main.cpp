/*
* Gökhan Yılmaz 
* METU Game Technologies Department 2017  
* 
* This code includes three viewshed algorithm R2, R3 and Van Kreveld's Algorithm.
* Van Kreveld's Algorithm is based on code provided by :
* Ferreira, C.R., et al., 2013. A Parallel Sweep Line Algorithm for
* Visibility Computation. In: Proceedings of the XIV Brazilian 
* Symposium on GeoInformatics (GeoInfo 2013), Campos do Jordão, SP,
* Brazil: 85-96.
*
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "npp.h"
#include "kernel.h"
#include "helper_cuda.h"

#include <windows.h>
#include "EasyBMP.h"
#include <vector>
#include <map>
#include <time.h>
#include "rbbst.h"
#include <iostream>
#include <fstream>





using namespace std;

double PCFreq = 0.0;
__int64 CounterStart = 0;

int radiusGlob = 0;
int currenIteration = 0;
int iterationCount = 10;
ofstream resultFile;
#define MEMORYMETRICS

unsigned long int SIZECONV = 1024 * 1024;



/**
* Starts the performance counter
**/
void startCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

/**
* Stops the performance counter and returns the result
**/
double getCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}


bool comparison(event_t a, event_t b) {
	if (a.angle < b.angle) return true;
	if (a.angle > b.angle) return false;

	return (a.type < b.type);
}


void writeViewshedToPicture(vs_t** viewshed, int ncols, int nrows, char* fileName)
{
	RGBApixel visibleColor;
	visibleColor.Red = 0;
	visibleColor.Green = 255;
	visibleColor.Blue = 0;

	RGBApixel notVisibleColor;
	notVisibleColor.Red = 255;
	notVisibleColor.Green = 0;
	notVisibleColor.Blue = 0;
	BMP img;
	img.SetSize(ncols, nrows);
	/** write output **/
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			if (viewshed[i][j] == 1)
			{
				img.SetPixel(j, i, visibleColor);
			}
			else
			{
				img.SetPixel(j, i, notVisibleColor);
			}
		}
	}
	img.WriteToFile(fileName);
}

void writeViewshedToPicture(vs_t* viewshed, int ncols, int nrows, char* fileName)
{
	RGBApixel visibleColor;
	visibleColor.Red = 0;
	visibleColor.Green = 255;
	visibleColor.Blue = 0;

	RGBApixel notVisibleColor;
	notVisibleColor.Red = 255;
	notVisibleColor.Green = 0;
	notVisibleColor.Blue = 0;
	BMP img;
	img.SetSize(ncols, nrows);
	for (int i = 0; i<nrows; i++)
	{
		for (int j = 0; j <ncols; j++)
		{
			int index = i*ncols + j;
			if (viewshed[index] == 1)
			{
				img.SetPixel(j, i, visibleColor);
			}
			else
			{
				img.SetPixel(j, i, notVisibleColor);
			}
		}
	}
	img.WriteToFile(fileName);
}

double runTestCPUVAN(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t** elev)
{

	vs_t** viewshed;
	elev_t observer_elev;
	RBTree* tree;


	observer_elev = elev[observerY][observerY] + observer_ht;

	/** alloc viewshed matrix **/
	viewshed = new vs_t*[nrows];
	for (int i = 0; i < nrows; i++)
		viewshed[i] = new vs_t[ncols];

	tree = new RBTree;



	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);


	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;
	int eventSize = ((width*height) * 3);

#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{
		unsigned long elevMemSize = sizeof(elev_t) * ncols*nrows / SIZECONV;
		unsigned long eventMemSize = (eventSize * sizeof(event_t)) / SIZECONV;
		unsigned long viewshedMemSize = ncols*nrows*sizeof(vs_t) / SIZECONV;

		std::cout << "CPU VAN Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Event List Memory Consumption(@CPU): " << eventMemSize << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize + eventMemSize << std::endl;


		resultFile << "CPU VAN Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Event List Memory Consumption(@CPU): " << eventMemSize << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize + eventMemSize << std::endl;
	}
	

#endif


	startCounter();

	vector<event_t> events;
	events.reserve(eventSize);

	for (int y = minY; y <= maxY; y++)
	{
		for (int x = minX; x <= maxX; x++)
		{
			if (x == observerX && y == observerY)
			{
				continue;
			}
			//if(x == observerX && y == ob
			int deltaY = y - observerY;
			int deltaX = x - observerX;
			double slope = double(deltaY) / deltaX;
			double angle = atan2((double)-deltaY, (double)deltaX);
			if (angle < 0) angle += 2 * M_PI;
			// calculate enter, center and exit angles depending on cell's quadrant
			double enterOffset[2];
			double exitOffset[2];

			if (deltaY < 0 && deltaX>0)
			{ // first quadrant
				enterOffset[0] = +0.5; enterOffset[1] = +0.5; exitOffset[0] = -0.5;	exitOffset[1] = -0.5;
			}
			else if (deltaY < 0 && deltaX < 0)
			{ // second quadrant
				enterOffset[0] = -0.5; enterOffset[1] = +0.5; exitOffset[0] = +0.5;	exitOffset[1] = -0.5;
			}
			else if (deltaY > 0 && deltaX < 0)
			{ // third quadrant
				enterOffset[0] = -0.5; enterOffset[1] = -0.5; exitOffset[0] = +0.5;	exitOffset[1] = +0.5;
			}
			else if (deltaY > 0 && deltaX>0)
			{ // fourth quadrant
				enterOffset[0] = +0.5; enterOffset[1] = -0.5; exitOffset[0] = -0.5;	exitOffset[1] = +0.5;
			}
			else if (deltaY < 0 && deltaX == 0)
			{ // to the north
				enterOffset[0] = +0.5; enterOffset[1] = +0.5; exitOffset[0] = +0.5; exitOffset[1] = -0.5;
			}
			else if (deltaY == 0 && deltaX < 0)
			{ // to the west
				enterOffset[0] = -0.5; enterOffset[1] = +0.5; exitOffset[0] = +0.5; exitOffset[1] = +0.5;
			}
			else if (deltaY > 0 && deltaX == 0)
			{ // to the south
				enterOffset[0] = -0.5; enterOffset[1] = -0.5; exitOffset[0] = -0.5;	exitOffset[1] = +0.5;
			}
			else if (deltaY == 0 && deltaX>0) { // to the east
				enterOffset[0] = +0.5; enterOffset[1] = -0.5; exitOffset[0] = -0.5; exitOffset[1] = -0.5;
			}

			event_t tmp;
			int dy = y - observerY;
			int dx = x - observerX;
			tmp.dist = dx*dx + dy*dy;

			// inserting ENTER event
			double Y = y + enterOffset[0];
			double X = x + enterOffset[1];
			double angleEnter = atan2(observerY - Y, X - observerX);
			if (angleEnter < 0) angleEnter += 2 * M_PI;

			if (y == observerY && x > observerX)
			{
				angleEnter = -1000;
			}

			tmp.type = ENTERING_EVENT;
			tmp.row = Y; tmp.col = X; tmp.angle = angleEnter;
			events.push_back(tmp);

			// inserting CENTER event
			tmp.angle = atan2((double)observerY - y, x - observerX);
			if (tmp.angle < 0) tmp.angle += 2 * M_PI;
			// just insert it if center is inside this this sector
			tmp.type = CENTER_EVENT;
			events.push_back(tmp);

			// inserting EXIT event
			tmp.type = EXITING_EVENT;
			Y = y + exitOffset[0]; X = x + exitOffset[1];
			double angleExit = atan2(observerY - Y, X - observerX);
			if (angleExit < 0) angleExit += 2 * M_PI;
			tmp.angle = angleExit;
			events.push_back(tmp);
		}
	}


	/** sort events **/
	sort(events.begin(), events.end(), comparison);

	//Process
	TreeValue tv;
	tv.gradient = SMALLEST_GRADIENT;
	tv.key = 0;
	tv.maxGradient = SMALLEST_GRADIENT;
	tree = create_tree(tv);

	std::vector<event_t>::iterator it;
	for (it = events.begin(); it != events.end(); ++it)
	{
		if (it->type == ENTERING_EVENT) {
			//calculate gradient
			double diff_elev = elev[it->row][it->col] - observer_elev;
			tv.gradient = (diff_elev*diff_elev) / it->dist;
			if (diff_elev < 0) tv.gradient *= -1;

			tv.key = it->dist;
			tv.maxGradient = SMALLEST_GRADIENT;

			insert_into(tree, tv);
		}
		else if (it->type == EXITING_EVENT) {
			delete_from(tree, it->dist);
		}
		else { // CENTER_EVENT 
			//calculate gradient
			double diff_elev = elev[it->row][it->col] + /*target_ht*/ -observer_elev;
			double gradient = (diff_elev*diff_elev) / it->dist;
			if (diff_elev < 0) gradient *= -1;

			double max_grad = find_max_gradient_within_key(tree, it->dist);
			if (max_grad < gradient) { // the cell is visible!
				viewshed[it->row][it->col] = 1;
			}
			else
			{
				viewshed[it->row][it->col] = 0;
			}
		}
	}

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputCPUVAN%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}


	delete tree;
	/** delete matrices */
	for (int i = 0; i<nrows; i++) {
		//delete[] elev[i];
		delete[] viewshed[i];
	}
	//delete[] elev;
	delete[] viewshed;



	//Return counter
	return result;
}

double runTestCPUR3(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t** elev)
{
	vs_t** viewshed;
	elev_t observer_elev;

	observer_elev = elev[observerY][observerY] + observer_ht;

	/** alloc viewshed matrix **/
	viewshed = new vs_t*[nrows];
	for (int i = 0; i<nrows; i++)
		viewshed[i] = new vs_t[ncols];

	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);


	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;


#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{
		long elevMemSize = sizeof(elev_t) * ncols*nrows / SIZECONV;
		long viewshedMemSize = ncols*nrows*sizeof(vs_t) / SIZECONV;

		std::cout << "CPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "CPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
	}

#endif

	startCounter();


	for (int y = minY; y <= maxY; y++)
	{
		for (int x = minX; x <= maxX; x++)
		{
			if (x == observerX && y == observerY)
			{
				viewshed[y][x] = 1;
			}

			int x1 = observerX;
			int y1 = observerY;
			int x2 = x;
			int y2 = y;

			int delta_x(x2 - x1);
			// if x1 == x2, then it does not matter what we set here
			signed char const ix((delta_x > 0) - (delta_x < 0));
			delta_x = std::abs(delta_x) << 1;

			int delta_y(y2 - y1);
			// if y1 == y2, then it does not matter what we set here
			signed char const iy((delta_y > 0) - (delta_y < 0));
			delta_y = std::abs(delta_y) << 1;


			float maxGradient = -10000;

			if (delta_x >= delta_y)
			{
				// error may go below zero
				int error(delta_y - (delta_x >> 1));

				while (x1 != x2)
				{
					if ((error >= 0) && (error || (ix > 0)))
					{
						error -= delta_x;
						y1 += iy;
					}
					// else do nothing

					error += delta_y;
					x1 += ix;


					int deltaY = y1 - observerY;
					int deltaX = x1 - observerX;
					float dist2 = deltaX*deltaX + deltaY*deltaY;

					double diff_elev = elev[y1][x1] - observer_elev;
					float gradient = (diff_elev*diff_elev) / dist2;
					if (diff_elev < 0) gradient *= -1;

					if (y1 == y && x1 == x)
					{
						if (gradient > maxGradient)
						{
							viewshed[y][x] = 1;
						}
						else
						{
							viewshed[y][x] = 0;
						}
					}
					else
					{
						if (gradient > maxGradient)
						{
							maxGradient = gradient;
						}
					}

				}
			}
			else
			{
				// error may go below zero
				int error(delta_x - (delta_y >> 1));

				while (y1 != y2)
				{
					if ((error >= 0) && (error || (iy > 0)))
					{
						error -= delta_y;
						x1 += ix;
					}
					// else do nothing

					error += delta_x;
					y1 += iy;

					int deltaY = y1 - observerY;
					int deltaX = x1 - observerX;
					float dist2 = deltaX*deltaX + deltaY*deltaY;

					double diff_elev = elev[y1][x1] - observer_elev;
					float gradient = (diff_elev*diff_elev) / dist2;
					if (diff_elev < 0) gradient *= -1;
					if (y1 == y && x1 == x)
					{
						if (gradient > maxGradient)
						{
							viewshed[y][x] = 1;
						}
						else
						{
							viewshed[y][x] = 0;
						}
					}
					else
					{
						if (gradient > maxGradient)
						{
							maxGradient = gradient;
						}
					}

				}
			}
		}
	}

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputCPUR3%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}

	/** delete matrices */
	for (int i = 0; i<nrows; i++) {
		//delete[] elev[i];
		delete[] viewshed[i];
	}
	//delete[] elev;
	delete[] viewshed;



	//Return counter
	return result;
}

void iterateLine(int x, int y, int observerX, int observerY, elev_t** elev, vs_t** viewshed, elev_t observer_elev)
{
	int x1 = observerX;
	int y1 = observerY;
	int x2 = x;
	int y2 = y;

	int delta_x(x2 - x1);
	// if x1 == x2, then it does not matter what we set here
	signed char const ix((delta_x > 0) - (delta_x < 0));
	delta_x = std::abs(delta_x) << 1;

	int delta_y(y2 - y1);
	// if y1 == y2, then it does not matter what we set here
	signed char const iy((delta_y > 0) - (delta_y < 0));
	delta_y = std::abs(delta_y) << 1;


	float maxGradient = -10000;

	if (delta_x >= delta_y)
	{
		// error may go below zero
		int error(delta_y - (delta_x >> 1));

		while (x1 != x2)
		{
			if ((error >= 0) && (error || (ix > 0)))
			{
				error -= delta_x;
				y1 += iy;
			}
			// else do nothing

			error += delta_y;
			x1 += ix;



			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX*deltaX + deltaY*deltaY;

			double diff_elev = elev[y1][x1] - observer_elev;
			float gradient = (diff_elev*diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[y1][x1] = 1;
			}
			else
			{
				viewshed[y1][x1] = 0;
			}
		}
	}
	else
	{
		// error may go below zero
		int error(delta_x - (delta_y >> 1));

		while (y1 != y2)
		{
			if ((error >= 0) && (error || (iy > 0)))
			{
				error -= delta_y;
				x1 += ix;
			}
			// else do nothing

			error += delta_x;
			y1 += iy;

			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX*deltaX + deltaY*deltaY;

			double diff_elev = elev[y1][x1] - observer_elev;
			float gradient = (diff_elev*diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[y1][x1] = 1;
			}
			else
			{
				viewshed[y1][x1] = 0;
			}

		}
	}
}

double runTestCPUR2(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t** elev)
{

	vs_t** viewshed;
	elev_t observer_elev;

	observer_elev = elev[observerY][observerY] + observer_ht;

	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);


	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;

	int x = minX;
	int y = minY;

#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{
		long elevMemSize = sizeof(elev_t) * ncols*nrows / SIZECONV;
		long viewshedMemSize = ncols*nrows*sizeof(vs_t) / SIZECONV;

		std::cout << "CPU R2 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "CPU R2 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
	}

#endif

	startCounter();

	/** alloc viewshed matrix **/
	viewshed = new vs_t*[nrows];
	for (int i = 0; i < nrows; i++)
		viewshed[i] = new vs_t[ncols];

	viewshed[observerY][observerX] = 1;

	for (int y = maxY, x = maxX; y >= minY; y--) { // right border (going up)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}
	for (int x = maxX, y = minY; x >= minX; x--) { // top border (going left)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}
	for (int y = minY, x = minX; y <= maxY; y++) { // left border (going down)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}
	for (int x = minX, y = maxY; x <= maxX; x++) { // bottom border (going right)
		iterateLine(x, y, observerX, observerY, elev, viewshed, observer_elev);
	}

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputCPUR2%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}
	
	/** delete matrices */
	for (int i = 0; i<nrows; i++) {
		//delete[] elev[i];
		delete[] viewshed[i];
	}
	//delete[] elev;
	delete[] viewshed;

	//Return counter
	return result;
}

double runTestGPUOnlyEventVAN(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t** elev)
{
	vs_t** viewshed;
	elev_t observer_elev;
	RBTree* tree;

	observer_elev = elev[observerY][observerY] + observer_ht;
	
	tree = new RBTree;



	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;
	long int eventSize = ((width*height) * 3);


#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{

		long elevMemSize = sizeof(elev_t) * ncols*nrows / SIZECONV;
		long eventMemSize = eventSize * sizeof(event_t) / SIZECONV;
		long viewshedMemSize = ncols*nrows*sizeof(vs_t) / SIZECONV;

		std::cout << "GPU VAN Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Event List Memory Consumption(@CPU): " << eventMemSize << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize + eventMemSize << std::endl;

		std::cout << "Event List Memory Consumption(@GPU): " << eventMemSize << std::endl;
		std::cout << "Total Memory Consumption(@GPU): " << eventMemSize << std::endl;

		resultFile << "GPU VAN Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Event List Memory Consumption(@CPU): " << eventMemSize << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize + eventMemSize << std::endl;

		resultFile << "Event List Memory Consumption(@GPU): " << eventMemSize << std::endl;
		resultFile << "Total Memory Consumption(@GPU): " << eventMemSize << std::endl;
	}

#endif

	startCounter();

	/** alloc viewshed matrix **/
	viewshed = new vs_t*[nrows];
	for (int i = 0; i < nrows; i++)
		viewshed[i] = new vs_t[ncols];



	event_t* events = new event_t[eventSize];
	event_t* d_events;
	cudaMalloc(reinterpret_cast<void **>(&d_events), sizeof(event_t)*eventSize);
	calculateEventsWrapper(d_events, minX, maxX, minY, maxY, observerX, observerY);
	cudaDeviceSynchronize();
	cudaError_t eType = cudaGetLastError();
	checkCudaErrors(eType);
	//astarWrapper(NULL, NULL, NULL, 5,5,3, NULL, NULL, MULTI_THREAD_PER_AGENT_EXTERNAL_DEV_FUNCS);

	sortEventsWithThrust(d_events, eventSize);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(events, d_events, sizeof(event_t)*eventSize, cudaMemcpyDeviceToHost));

	//std::vector<event_t> events (eventsArr, eventsArr + eventSize);
	/** sort events **/
	//sort(events.begin(),events.end(),comparison);

	//return 0.5;
	//Process
	TreeValue tv;
	tv.gradient = SMALLEST_GRADIENT;
	tv.key = 0;
	tv.maxGradient = SMALLEST_GRADIENT;
	tree = create_tree(tv);

	std::vector<event_t>::iterator it;
	for (int i = 0; i < eventSize; i++)
	{
		event_t it = events[i];
		if (it.row == observerY && it.col == observerX)
		{
			continue;
		}

		if (it.type == ENTERING_EVENT) {
			//calculate gradient
			double diff_elev = elev[it.row][it.col] - observer_elev;
			tv.gradient = (diff_elev*diff_elev) / it.dist;
			if (diff_elev < 0) tv.gradient *= -1;

			tv.key = it.dist;
			tv.maxGradient = SMALLEST_GRADIENT;

			insert_into(tree, tv);
		}
		else if (it.type == EXITING_EVENT) {
			delete_from(tree, it.dist);
		}
		else { // CENTER_EVENT 
			//calculate gradient
			double diff_elev = elev[it.row][it.col] + /*target_ht*/ -observer_elev;
			double gradient = (diff_elev*diff_elev) / it.dist;
			if (diff_elev <0) gradient *= -1;

			double max_grad = find_max_gradient_within_key(tree, it.dist);
			if (max_grad < gradient) { // the cell is visible!
				viewshed[it.row][it.col] = 1;
			}
			else
			{
				viewshed[it.row][it.col] = 0;
			}
		}
	}

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputGPUVAN%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}

	delete tree;
	/** delete matrices */
	for (int i = 0; i<nrows; i++) {
		//delete[] elev[i];
		delete[] viewshed[i];
	}
	//delete[] elev;
	delete[] viewshed;
	delete[] events;

	checkCudaErrors(cudaFree(d_events));
	//Return counter
	return result;
}

double test(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t* elev)
{
	elev_t observer_elev;
	RBTree* tree;
	observer_elev = elev[observerY*ncols + observerX] + observer_ht;
	tree = new RBTree;

	startCounter();

	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;
	int eventSize = ((width*height) * 3);



	width -= 1;
	height -= 1;

	int viewshedSize = width*height;
	vs_t* viewshed = 0;
	viewshed = new vs_t[viewshedSize];
	vs_t* d_viewshed = 0;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_viewshed), sizeof(vs_t)*viewshedSize));

	testWrapper(d_viewshed, viewshedSize);

	checkCudaErrors(cudaMemcpy(viewshed, d_viewshed, sizeof(vs_t)*viewshedSize, cudaMemcpyDeviceToHost));

	if (currenIteration == iterationCount)
	{
		writeViewshedToPicture(viewshed, ncols, nrows, "test.bmp");
	}

	return 5;

}

double runTestGPUKernel(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t* elev)
{

	elev_t observer_elev;
	RBTree* tree;

	observer_elev = elev[observerY*ncols + observerX] + observer_ht;

	tree = new RBTree;

	startCounter();
	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;
	int eventSize = ((width*height) * 3);

	event_t* events = new event_t[eventSize];
	event_t* d_events;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_events), sizeof(event_t)*eventSize));
	calculateEventsWrapper(d_events, minX, maxX, minY, maxY, observerX, observerY);
	cudaError_t eType = cudaGetLastError();
	checkCudaErrors(eType);
	//astarWrapper(NULL, NULL, NULL, 5,5,3, NULL, NULL, MULTI_THREAD_PER_AGENT_EXTERNAL_DEV_FUNCS);

	sortEventsWithThrust(d_events, eventSize);

	//checkCudaErrors(cudaMemcpy(events, d_events, sizeof(event_t)*eventSize,cudaMemcpyDeviceToHost));

	//std::vector<event_t> events (eventsArr, eventsArr + eventSize);
	/** sort events **/
	//sort(events.begin(),events.end(),comparison);


	vs_t* viewshed = new vs_t[nrows*ncols];
	vs_t* d_viewshed;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_viewshed), sizeof(vs_t)*nrows*ncols));

	//checkCudaErrors(cudaMemset(&d_viewshed, 0,nrows*ncols));



	elev_t* d_elev;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_elev), sizeof(elev_t)*nrows*ncols));

	checkCudaErrors(cudaMemcpy(d_elev, elev, sizeof(elev_t)*nrows*ncols, cudaMemcpyHostToDevice));

	iterateOverEventsWrapper(d_events, d_elev, d_viewshed, observerX, observerY, eventSize, ncols, observer_elev);

	checkCudaErrors(cudaMemcpy(viewshed, d_viewshed, sizeof(vs_t)*nrows*ncols, cudaMemcpyDeviceToHost));

	//Get the result counter
	double result = getCounter();


	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputGPUKernel.%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}

	delete tree;
	/** delete matrices */

	//delete[] elev;
	delete[] viewshed;
	delete[] events;

	checkCudaErrors(cudaFree(d_events));
	checkCudaErrors(cudaFree(d_elev));
	checkCudaErrors(cudaFree(d_viewshed));
	//Return counter
	return result;
}

double runTestGPUEventAndTree(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t** elev)
{
	vs_t** viewshed;
	elev_t observer_elev;
	RBTree* tree;

	observer_elev = elev[observerY][observerY] + observer_ht;

	/** alloc viewshed matrix **/
	viewshed = new vs_t*[nrows];
	for (int i = 0; i<nrows; i++)
		viewshed[i] = new vs_t[ncols];

	tree = new RBTree;

	startCounter();

	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;
	int eventSize = ((width*height) * 3);


	event_t* events = new event_t[eventSize];
	event_t* d_events;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_events), sizeof(event_t)*eventSize));
	calculateEventsWrapper(d_events, minX, maxX, minY, maxY, observerX, observerY);
	cudaError_t eType = cudaGetLastError();
	checkCudaErrors(eType);
	//astarWrapper(NULL, NULL, NULL, 5,5,3, NULL, NULL, MULTI_THREAD_PER_AGENT_EXTERNAL_DEV_FUNCS);

	sortEventsWithThrust(d_events, eventSize);

	checkCudaErrors(cudaMemcpy(events, d_events, sizeof(event_t)*eventSize, cudaMemcpyDeviceToHost));

	//std::vector<event_t> events (eventsArr, eventsArr + eventSize);
	/** sort events **/
	//sort(events.begin(),events.end(),comparison);

	//return 0.5;
	//Process
	TreeValue tv;
	tv.gradient = SMALLEST_GRADIENT;
	tv.key = 0;
	tv.maxGradient = SMALLEST_GRADIENT;
	tree = create_tree(tv);

	treeVal* values = new treeVal[radius];
	resizeDeviceVector((int)(sqrt((float)(2 * radius*radius)) * 2));
	int currentSize = 0;
	int totalSize = 0;

	std::vector<event_t>::iterator it;
	for (int i = 0; i < eventSize; i++)
	{
		event_t it = events[i];
		if (it.row == observerY && it.col == observerX)
		{
			continue;
		}

		if (it.type == ENTERING_EVENT) {
			//calculate gradient
			double diff_elev = elev[it.row][it.col] - observer_elev;
			float gradient = (diff_elev*diff_elev) / it.dist;
			if (diff_elev < 0) tv.gradient *= -1;

			treeVal val;
			val.distance = it.dist;
			val.gradient = gradient;
			values[currentSize] = val;

			currentSize++;

			tv.key = it.dist;
			tv.maxGradient = SMALLEST_GRADIENT;

			//insert_into(tree, tv);

		}
		else if (it.type == EXITING_EVENT) {
			//delete_from(tree,it.dist);

			if (currentSize > 0)
			{
				insertToEventValues(values, currentSize, totalSize);
				totalSize += currentSize;
				currentSize = 0;
			}

			if (totalSize >  0)
			{
				int check = totalSize;
				deletFromTreeWithThrust(it.dist, totalSize);
				/*if(check != totalSize)
				{
				int a =5;
				a++;
				}*/
			}
		}
		else { // CENTER_EVENT 
			//calculate gradient
			if (currentSize > 0)
			{
				insertToEventValues(values, currentSize, totalSize);
				totalSize += currentSize;
				currentSize = 0;
			}

			if (totalSize > 0)
			{
				double diff_elev = elev[it.row][it.col] + /*target_ht*/ -observer_elev;
				double gradient = (diff_elev*diff_elev) / it.dist;
				if (diff_elev <0) gradient *= -1;

				//sortTreeWithThrust(totalSize);			
				treeVal findVal;
				findMaxInTreeUnsorted(totalSize, gradient, it.dist, findVal);
				if (findVal.gradient < -600)
				{
					viewshed[it.row][it.col] = 1;

				}
				else
				{
					viewshed[it.row][it.col] = 0;
				}

			}


			//double max_grad = find_max_gradient_within_key(tree,it.dist);
			//if ( max_grad < gradient ) { // the cell is visible!
			//viewshed[it.row][it.col] = 1;
			//}
			//else
			//{
			//viewshed[it.row][it.col] = 0;
			//}
		}
	}

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputGPU2%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}

	delete tree;
	/** delete matrices */
	for (int i = 0; i<nrows; i++) {
		//delete[] elev[i];
		delete[] viewshed[i];
	}
	//delete[] elev;
	delete[] viewshed;
	delete[] events;
	delete[] values;

	checkCudaErrors(cudaFree(d_events));
	//Return counter
	return result;
}

double runTestGPUKernelPartial(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t* elev)
{
	elev_t observer_elev;
	RBTree* tree;

	observer_elev = elev[observerY*ncols + observerX] + observer_ht;

	tree = new RBTree;

	startCounter();

	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;
	int eventSize = ((width*height) * 3);

	width -= 1;
	height -= 1;

	event_t* events = new event_t[eventSize];
	event_t* d_events;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_events), sizeof(event_t)*eventSize));
	calculateEventsWrapper(d_events, minX, maxX, minY, maxY, observerX, observerY);
	cudaError_t eType = cudaGetLastError();
	checkCudaErrors(eType);

	sortEventsWithThrust(d_events, eventSize);

	int viewShedSize = width*height;
	vs_t* viewshed = new vs_t[viewShedSize];
	vs_t* d_viewshed;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_viewshed), sizeof(vs_t)*viewShedSize));

	elev_t* d_elev;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_elev), sizeof(elev_t)*nrows*ncols));

	checkCudaErrors(cudaMemcpy(d_elev, elev, sizeof(elev_t)*nrows*ncols, cudaMemcpyHostToDevice));

	iterateOverEventsWrapperPartialViewshed(d_events, d_elev, d_viewshed, observerX, observerY, eventSize, ncols, observer_elev, minX, minY, width);

	checkCudaErrors(cudaMemcpy(viewshed, d_viewshed, sizeof(vs_t)*viewShedSize, cudaMemcpyDeviceToHost));

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputGPUKernelPartial%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}

	delete tree;
	/** delete matrices */

	//delete[] elev;
	delete[] viewshed;
	delete[] events;

	checkCudaErrors(cudaFree(d_events));
	checkCudaErrors(cudaFree(d_elev));
	checkCudaErrors(cudaFree(d_viewshed));
	//Return counter
	return result;
}

double runTestGPUR3(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t* elev)
{
	elev_t observer_elev;

	observer_elev = elev[observerY*ncols + observerX] + observer_ht;

	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

#ifdef MEMORYMETRICS

	if (currenIteration == 1)
	{

		long elevMemSize = sizeof(elev_t) * ncols*nrows / SIZECONV;
		long viewshedMemSize = ncols*nrows*sizeof(vs_t) / SIZECONV;

		std::cout << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;
	}

#endif

	startCounter();


	vs_t* viewshed = new vs_t[nrows*ncols];
	vs_t* d_viewshed;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_viewshed), sizeof(vs_t)*nrows*ncols));
	elev_t* d_elev;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_elev), sizeof(elev_t)*nrows*ncols));
	checkCudaErrors(cudaMemcpy(d_elev, elev, sizeof(elev_t)*nrows*ncols, cudaMemcpyHostToDevice));
	cudaR3Wrapper(d_viewshed, d_elev, observer_elev, minX, maxX, minY, maxY, observerX, observerY, ncols);
	checkCudaErrors(cudaMemcpy(viewshed, d_viewshed, sizeof(vs_t)*nrows*ncols, cudaMemcpyDeviceToHost));

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputGPUR3%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}

	delete[] viewshed;
	checkCudaErrors(cudaFree(d_elev));
	checkCudaErrors(cudaFree(d_viewshed));
	//Return counter
	return result;
}

double runTestGPUR2(int nrows, int ncols, int radius, int observerX, int observerY, int observer_ht, elev_t* elev)
{
	elev_t observer_elev;	
	observer_elev = elev[observerY*ncols + observerX] + observer_ht;

	//CALCULATE EVENTS
	int minY = max(0, observerY - radius);
	int maxY = min(nrows - 1, observerY + radius);
	int minX = max(0, observerX - radius);
	int maxX = min(ncols - 1, observerX + radius);

#ifdef MEMORYMETRICS
	if (currenIteration == 1)
	{

		long elevMemSize = sizeof(elev_t) * ncols*nrows / SIZECONV;
		long viewshedMemSize = ncols*nrows*sizeof(vs_t) / SIZECONV;

		std::cout << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		std::cout << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		std::cout << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		std::cout << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;


		resultFile << "GPU R3 Memory Consumption for " << nrows << "*" << ncols << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@CPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@CPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@CPU): " << viewshedMemSize + elevMemSize << std::endl;
		resultFile << "Elevation Data  Memory Consumption(@GPU): " << elevMemSize << std::endl;
		resultFile << "Viewshed Memory Consumption(@GPU): " << viewshedMemSize << std::endl;
		resultFile << "Total Memory Consumption(@GPU): " << viewshedMemSize + elevMemSize << std::endl;
	}
#endif

	startCounter();
	vs_t* viewshed = new vs_t[nrows*ncols];
	vs_t* d_viewshed;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_viewshed), sizeof(vs_t)*nrows*ncols));
	elev_t* d_elev;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_elev), sizeof(elev_t)*nrows*ncols));
	checkCudaErrors(cudaMemcpy(d_elev, elev, sizeof(elev_t)*nrows*ncols, cudaMemcpyHostToDevice));
	cudaR2Wrapper(d_viewshed, d_elev, observer_elev, minX, maxX, minY, maxY, observerX, observerY, ncols);
	checkCudaErrors(cudaMemcpy(viewshed, d_viewshed, sizeof(vs_t)*nrows*ncols, cudaMemcpyDeviceToHost));

	//Get the result counter
	double result = getCounter();

	if (currenIteration == iterationCount)
	{
		char buffer[100];
		sprintf(buffer, "outputGPUR2%d.bmp", radiusGlob);
		writeViewshedToPicture(viewshed, ncols, nrows, buffer);
	}

	delete[] viewshed;
	checkCudaErrors(cudaFree(d_elev));
	checkCudaErrors(cudaFree(d_viewshed));
	//Return counter
	return result;
}

// Main function.
int
main(int argc, char ** argv)
{

	//Seed random
	srand(time(NULL));


	int observer_ht = 15;

	double result;


	int nrowCounts[6] = { 5000, 10000, 15000, 20000, 25000, 30000 };
	int nColumnCounts[6] = { 5000, 10000, 15000, 20000, 25000, 30000 };
	int observerXs[6] = { 2499, 4999, 7499, 9999, 12499, 14999 };
	int observerYs[6] = { 2499, 4999, 7499, 9999, 12499, 14999 };
	//int radiuses[6] = { 2378, 2000, 3500, 4500, 6000, 7000};
	//int radiuses[6] = { 2350, 4950, 7450, 9950, 12450, 14950 };
	int radiuses[6] = { 2350, 3127, 7450, 9950, 12450, 14950 };


	//ASSIGN ELEVATION DATAS HERE
	char* elevPaths[6] =
	{
		"E:\\gokhan_workspace\\terrain_data\\terrainR2_5000.hgt"
		, "E:\\gokhan_workspace\\terrain_data\\terrainR2_10000.hgt"
		, "E:\\gokhan_workspace\\terrain_data\\terrainR2_15000.hgt"
		, "E:\\gokhan_workspace\\terrain_data\\terrainR2_20000.hgt"
		, "E:\\gokhan_workspace\\terrain_data\\terrainR2_25000.hgt"
		, "E:\\gokhan_workspace\\terrain_data\\terrainR2_30000.hgt"
	};




	double resultCPUVAN = 0;
	double resultGPUVAN = 0;

	double resultCPUR2 = 0;
	double resultGPUR2 = 0;

	double resultCPUR3 = 0;
	double resultGPUR3 = 0;

	resultFile.open("results.txt");

	for (int i = 0; i < 6; i++)
	{
		radiusGlob = radiuses[i];
		elev_t** elev;
		//Read elevation
		FILE* f = fopen(elevPaths[i], "rb");
		elev = new elev_t*[nrowCounts[i]];
		for (int k = 0; k < nrowCounts[i]; k++) {
			elev[k] = new elev_t[nColumnCounts[i]];
			fread(reinterpret_cast<char*>(elev[k]), sizeof(elev_t), nColumnCounts[i], f);
		}
		fclose(f);


		elev_t* elev1D = new elev_t[nrowCounts[i] * nColumnCounts[i]];
		//Read elevation
		f = fopen(elevPaths[i], "rb");
		int readCount = 0;
		for (int k = 0; k < nrowCounts[i]; k++) {
			fread(reinterpret_cast<char*>(elev1D + readCount), sizeof(elev_t), nColumnCounts[i], f);
			readCount += nColumnCounts[i];
		}

		fclose(f);

		resultCPUVAN = 0;
		resultGPUVAN = 0;

		resultCPUR2 = 0;
		resultGPUR2 = 0;

		resultCPUR3 = 0;
		resultGPUR3 = 0;

		event_t* d_events;
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_events), sizeof(event_t)*25000*25000*3));
		getchar();


		for (int j = 0; j < iterationCount; j++)
		{
			currenIteration = j + 1;
			double result;
			//VAN
			std::cout << "Running test for CPU VAN with I " << i << std::endl;
			resultFile << "Running test for CPU VAN with I " << i << std::endl;
			result = runTestCPUVAN(nrowCounts[i], nColumnCounts[i], radiuses[i], observerXs[i], observerYs[i], observer_ht, elev);
			std::cout << "CPU VAN test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultFile << "CPU VAN test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultCPUVAN += result;

			std::cout << "Running test for GPU VAN with I " << i << std::endl;
			resultFile << "Running test for GPU VAN with I " << i << std::endl;;
			result = runTestGPUOnlyEventVAN(nrowCounts[i], nColumnCounts[i], radiuses[i], observerXs[i], observerYs[i], observer_ht, elev);
			std::cout << "GPU VAN test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultFile << "GPU VAN test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultGPUVAN += result;

			////R2
			std::cout << "Running test for CPU R2 with I " << i << std::endl;;
			resultFile << "Running test for CPU R2 with I " << i << std::endl;;
			result = runTestCPUR2(nrowCounts[i], nColumnCounts[i], radiuses[i], observerXs[i], observerYs[i], observer_ht, elev);
			std::cout << "CPU R2 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultFile << "CPU R2 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultCPUR2 += result;

			std::cout << "Running test for GPU R2 with I " << i << std::endl;;
			resultFile << "Running test for GPU R2 with I " << i << std::endl;;
			result = runTestGPUR2(nrowCounts[i], nColumnCounts[i], radiuses[i], observerXs[i], observerYs[i], observer_ht, elev1D);
			std::cout << "GPU R2 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultFile << "GPU R2 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultGPUR2 += result;

			////R3
			std::cout << "Running test for CPU R3 with I " << i << std::endl;;
			resultFile << "Running test for CPU R3 with I " << i << std::endl;;
			result = runTestCPUR3(nrowCounts[i], nColumnCounts[i], radiuses[i], observerXs[i], observerYs[i], observer_ht, elev);
			std::cout << "CPU R3 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultFile << "CPU R3 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultCPUR3 += result;


			std::cout << "Running test for GPU R3 with I " << i << std::endl;;
			resultFile << "Running test for GPU R3 with I " << i << std::endl;;
			result = runTestGPUR3(nrowCounts[i], nColumnCounts[i], radiuses[i], observerXs[i], observerYs[i], observer_ht, elev1D);
			std::cout << "GPU R3 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultFile << "GPU R3 test completed. Result is " << result << std::endl << std::endl << std::endl;
			resultGPUR3 += result;


			
		}

		std::cout << std::endl << std::endl << std::endl;
		resultFile << std::endl << std::endl << std::endl;
		std::cout << "Results for " << nrowCounts[i] << "*" << nColumnCounts[i] << std::endl;
		resultFile << "Results for " << nrowCounts[i] << "*" << nColumnCounts[i] << std::endl;

		std::cout << "CPU VAN average result is " << resultCPUVAN / iterationCount << std::endl;
		std::cout << "GPU VAN average result is " << resultGPUVAN / iterationCount << std::endl;
		resultFile << "CPU VAN average result is " << resultCPUVAN / iterationCount << std::endl;
		resultFile << "GPU VAN average result is " << resultGPUVAN / iterationCount << std::endl;

		std::cout << "CPU R2 average result is " << resultCPUR2 / iterationCount << std::endl ;
		std::cout << "GPU R2 average result is " << resultGPUR2 / iterationCount << std::endl ;
		resultFile << "CPU R2 average result is " << resultCPUR2 / iterationCount << std::endl;
		resultFile << "GPU R2 average result is " << resultGPUR2 / iterationCount << std::endl ;


		std::cout << "CPU R3 average result is " << resultCPUR3 / iterationCount << std::endl;
		std::cout << "GPU R3 average result is " << resultGPUR3 / iterationCount << std::endl;
		resultFile << "CPU R3 average result is " << resultCPUR3 / iterationCount << std::endl;
		resultFile << "GPU R3 average result is " << resultGPUR3 / iterationCount << std::endl;

		delete elev1D;

		for (int k = 0; k < nrowCounts[i]; k++) {
			delete[] elev[k];
		}
		delete[] elev;

	}



	resultFile.close();
	cout << "Finished" << endl;
	std::getchar();
	cudaDeviceReset();
	return 0;
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )
