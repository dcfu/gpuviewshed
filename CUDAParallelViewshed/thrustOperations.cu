#include "kernel.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/sort.h>


struct EventCmp {
	__host__ __device__
		bool operator()(const event_t& a, const event_t& b) {
			if (a.angle < b.angle) return true;
			if (a.angle > b.angle) return false;

			return (a.type < b.type);
	}
};

struct TreeCmp {
	__host__ __device__
		bool operator()(const treeVal& a, const treeVal& b) {
			if (a.distance < b.distance) return true;
			if (a.distance > b.distance) return false;

			return true;
	}
};

struct RemoveTree
   {
     float dist;

     RemoveTree(float _dist) : dist(_dist) {};

     __host__ __device__
     bool operator()(const treeVal& a) const  {

		 return fabs(a.distance - dist) < 0.0000001;
		
     }
 };
   

struct FindTree
   {
     float dist;
	 float grad;

     FindTree(float _dist, float _grad) : dist(_dist), grad(grad) {};

     __host__ __device__
     bool operator()(const treeVal& a) const  {

		 if(a.distance < dist && a.gradient > grad)
		 {
			 return true;
		 }		
		 return false;
     }
 };


thrust::device_vector<treeVal> deviceValues;
//int currentSize = 0;

void resizeDeviceVector(int size)
{
	deviceValues.resize(size);
}



void sortTreeWithThrust(int totalSize)
{
	thrust::sort(deviceValues.begin(),deviceValues.begin()+totalSize, TreeCmp());
}

//void insertToEventValues(thrust::host_vector<treeVal>& values, int size)
void insertToEventValues(treeVal*   values, int size, int currentSize)
{
	thrust::copy(values, values + size, deviceValues.begin() + currentSize);
	//currentSize+=size;
	//thrust::device_vector<treeVal>::insert(devicesValues.begin(), size,values);
	//deviceValues.insert(devicesValues.begin(), size,values);
	
	//deviceValues.insert(deviceValues.begin(), size, values);	
}

void deletFromTreeWithThrust(float dist, int& totalSize)
{
	thrust::device_vector<treeVal>::iterator newEnd;
	newEnd = thrust::remove_if(deviceValues.begin(),deviceValues.begin()+totalSize, RemoveTree(dist));
	totalSize = thrust::distance(deviceValues.begin(), newEnd);
}


void findMaxInTreeUnsorted(int totalSize, float grad, float dist, treeVal& val)
{
	thrust::device_vector<treeVal>::iterator ret;

	ret = thrust::find_if(deviceValues.begin(),deviceValues.begin()+totalSize, FindTree(dist, grad));

	if(ret != deviceValues.begin()+totalSize)
	{
		treeVal findVal = *ret;
		val.distance = findVal.distance;
		val.gradient = findVal.gradient;
	}
	else
	{
		val.gradient = -1000;
	}
}

void sortEventsWithThrust(event_t* events, long int eventSize)
{
	thrust::device_ptr<event_t> dev_ptr(events);	
	thrust::sort(dev_ptr,dev_ptr + eventSize, EventCmp());	
}
