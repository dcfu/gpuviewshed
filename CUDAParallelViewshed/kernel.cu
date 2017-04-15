#include <cuda.h>
#include "npp.h"
#include <stdio.h>
#include "kernel.h"
#include <math.h>

#define BLOCK_DIM 512

#define EPSILON 0.0000001


/*<--------------------------------->
   //Private below this line */

/*find the max value in the given tree
   //you need to provide a compare function to compare the nodes */
__device__ float find_max_valueD(TreeNodeD * root)
{
    if (!root)
	return SMALLEST_GRADIENTD;    
    /*assert(root->value.maxGradient != SMALLEST_GRADIENTD);
       //LT: this shoudl be fixed
       //if (root->value.maxGradient != SMALLEST_GRADIENTD) */
    return root->value.maxGradient;
}

__device__ void left_rotateD(TreeNodeD ** root, TreeNodeD * x, TreeNodeD* NIL)
{
    TreeNodeD *y;

    y = x->right;

    /*maintain augmentation */
    double tmpMax;

    /*fix x */
    tmpMax = x->left->value.maxGradient > y->left->value.maxGradient ?
	x->left->value.maxGradient : y->left->value.maxGradient;

    if (tmpMax > x->value.gradient)
	x->value.maxGradient = tmpMax;
    else
	x->value.maxGradient = x->value.gradient;


    /*fix y */
    tmpMax = x->value.maxGradient > y->right->value.maxGradient ?
	x->value.maxGradient : y->right->value.maxGradient;

    if (tmpMax > y->value.gradient)
	y->value.maxGradient = tmpMax;
    else
	y->value.maxGradient = y->value.gradient;

    /*left rotation
       //see pseudocode on page 278 in CLRS */

    x->right = y->left;		/*turn y's left subtree into x's right subtree */
    y->left->parent = x;

    y->parent = x->parent;	/*link x's parent to y */

    if (x->parent == NIL) {
	*root = y;
    }
    else {
	if (x == x->parent->left)
	    x->parent->left = y;
	else
	    x->parent->right = y;
    }

    y->left = x;
    x->parent = y;

    return;
}

__device__ void right_rotateD(TreeNodeD ** root, TreeNodeD * y, TreeNodeD* NIL)
{
    TreeNodeD *x;

    x = y->left;

    /*maintain augmentation
       //fix y */
    double tmpMax;

    tmpMax = x->right->value.maxGradient > y->right->value.maxGradient ?
	x->right->value.maxGradient : y->right->value.maxGradient;

    if (tmpMax > y->value.gradient)
	y->value.maxGradient = tmpMax;
    else
	y->value.maxGradient = y->value.gradient;

    /*fix x */
    tmpMax = x->left->value.maxGradient > y->value.maxGradient ?
	x->left->value.maxGradient : y->value.maxGradient;

    if (tmpMax > x->value.gradient)
	x->value.maxGradient = tmpMax;
    else
	x->value.maxGradient = x->value.gradient;

    /*ratation */
    y->left = x->right;
    x->right->parent = y;

    x->parent = y->parent;

    if (y->parent == NIL) {
	*root = x;
    }
    else {
	if (y->parent->left == y)
	    y->parent->left = x;
	else
	    y->parent->right = x;
    }

    x->right = y;
    y->parent = x;

    return;
}


/*fix the rb tree after deletion */
__device__ void rb_delete_fixupD(TreeNodeD ** root, TreeNodeD * x, TreeNodeD* NIL)
{
    TreeNodeD *w;

    while (x != *root && x->color == RB_BLACKD) {
	if (x == x->parent->left) {
	    w = x->parent->right;
	    if (w->color == RB_REDD) {
		w->color = RB_BLACKD;
		x->parent->color = RB_REDD;
		left_rotateD(root, x->parent, NIL);
		w = x->parent->right;
	    }

	    if (w == NIL) {
		x = x->parent;
		continue;
	    }

	    if (w->left->color == RB_BLACKD && w->right->color == RB_BLACKD) {
		w->color = RB_REDD;
		x = x->parent;
	    }
	    else {
		if (w->right->color == RB_BLACKD) {
		    w->left->color = RB_BLACKD;
		    w->color = RB_REDD;
		    right_rotateD(root, w, NIL);
		    w = x->parent->right;
		}

		w->color = x->parent->color;
		x->parent->color = RB_BLACKD;
		w->right->color = RB_BLACKD;
		left_rotateD(root, x->parent, NIL);
		x = *root;
	    }

	}
	else {			/*(x==x->parent->right) */
	    w = x->parent->left;
	    if (w->color == RB_REDD) {
		w->color = RB_BLACKD;
		x->parent->color = RB_REDD;
		right_rotateD(root, x->parent, NIL);
		w = x->parent->left;
	    }

	    if (w == NIL) {
		x = x->parent;
		continue;
	    }

	    if (w->right->color == RB_BLACKD && w->left->color == RB_BLACKD) {
		w->color = RB_REDD;
		x = x->parent;
	    }
	    else {
		if (w->left->color == RB_BLACKD) {
		    w->right->color = RB_BLACKD;
		    w->color = RB_REDD;
		    left_rotateD(root, w, NIL);
		    w = x->parent->left;
		}

		w->color = x->parent->color;
		x->parent->color = RB_BLACKD;
		w->left->color = RB_BLACKD;
		right_rotateD(root, x->parent, NIL);
		x = *root;
	    }

	}
    }
    x->color = RB_BLACKD;

    return;
}


__device__ void init_nil_nodeD(RBTreeD* rbtree)
{
    if (rbtree->NIL != NULL) return;

    rbtree->NIL = (TreeNodeD *) malloc(sizeof(TreeNodeD));

    rbtree->NIL->color = RB_BLACKD;
    rbtree->NIL->value.gradient = SMALLEST_GRADIENTD;
    rbtree->NIL->value.maxGradient = SMALLEST_GRADIENTD;

    rbtree->NIL->parent = NULL;
    rbtree->NIL->left = NULL;
    rbtree->NIL->right = NULL;
    return;
}

/*you can write change this compare function, depending on your TreeValueD struct
   //compare function used by findMaxValue
   //-1: v1 < v2
   //0:  v1 = v2
   //2:  v1 > v2 */
__device__ char compare_valuesD(TreeValueD * v1, TreeValueD * v2)
{
    if (v1->gradient > v2->gradient)
	return 1;
    if (v1->gradient < v2->gradient)
	return -1;

    return 0;
}


/*a function used to compare two doubles */
__device__ char compare_doubleD(double a, double b)
{
    if (fabs(a - b) < EPSILON)
	return 0;
    if (a - b < 0)
	return -1;

    return 1;
}



/*create a tree node */
__device__ TreeNodeD *create_tree_nodeD(TreeValueD value, TreeNodeD* NIL)
{
    TreeNodeD *ret;


    ret = (TreeNodeD *) malloc(sizeof(TreeNodeD));


    ret->color = RB_REDD;

    ret->left = NIL;
    ret->right = NIL;
    ret->parent = NIL;

    ret->value = value;
    ret->value.maxGradient = SMALLEST_GRADIENTD;
    return ret;
}

__device__ void rb_insert_fixupD(TreeNodeD ** root, TreeNodeD * z, TreeNodeD* NIL)
{
    /*see pseudocode on page 281 in CLRS */
    TreeNodeD *y;

    while (z->parent->color == RB_REDD) {
	if (z->parent == z->parent->parent->left) {
	    y = z->parent->parent->right;
	    if (y->color == RB_REDD) {	/*case 1 */
		z->parent->color = RB_BLACKD;
		y->color = RB_BLACKD;
		z->parent->parent->color = RB_REDD;
		z = z->parent->parent;
	    }
	    else {
		if (z == z->parent->right) {	/*case 2 */
		    z = z->parent;
		    left_rotateD(root, z, NIL);	/*convert case 2 to case 3 */
		}
		z->parent->color = RB_BLACKD;	/*case 3 */
		z->parent->parent->color = RB_REDD;
		right_rotateD(root, z->parent->parent, NIL);
	    }

	}
	else {			/*(z->parent == z->parent->parent->right) */
	    y = z->parent->parent->left;
	    if (y->color == RB_REDD) {	/*case 1 */
		z->parent->color = RB_BLACKD;
		y->color = RB_BLACKD;
		z->parent->parent->color = RB_REDD;
		z = z->parent->parent;
	    }
	    else {
		if (z == z->parent->left) {	/*case 2 */
		    z = z->parent;
		    right_rotateD(root, z, NIL);	/*convert case 2 to case 3 */
		}
		z->parent->color = RB_BLACKD;	/*case 3 */
		z->parent->parent->color = RB_REDD;
		left_rotateD(root, z->parent->parent, NIL);
	    }
	}
    }
    (*root)->color = RB_BLACKD;

    return;
}


/*create node with its value set to the value given
   //and insert the node into the tree
   //rbInsertFixup may change the root pointer, so TreeNodeD** is passed in */
__device__ void insert_into_treeD(TreeNodeD ** root, TreeValueD value, TreeNodeD* NIL)
{
    TreeNodeD *curNode;
    TreeNodeD *nextNode;

    curNode = *root;

    if (compare_doubleD(value.key, curNode->value.key) == -1) {
	nextNode = curNode->left;
    }
    else {
	nextNode = curNode->right;
    }


    while (nextNode != NIL) {
	curNode = nextNode;

	int comp = compare_doubleD(value.key, curNode->value.key);
	
	if (comp == -1) {
	    nextNode = curNode->left;
	}
	else if (comp == 1) {
	    nextNode = curNode->right;
	}
	else {
	  //cerr << "Node already exists! Cancelling..." << endl;
	  return;
	}
	
    }

    /*create a new node 
       //and place it at the right place
       //created node is RED by default */
    nextNode = create_tree_nodeD(value, NIL);

    nextNode->parent = curNode;

    if (compare_doubleD(value.key, curNode->value.key) == -1) {
	curNode->left = nextNode;
    }
    else {
	curNode->right = nextNode;
    }

    TreeNodeD *inserted = nextNode;

    /*update augmented maxGradient */
    nextNode->value.maxGradient = nextNode->value.gradient;
    while (nextNode->parent != NIL) {
	if (nextNode->parent->value.maxGradient < nextNode->value.maxGradient)
	    nextNode->parent->value.maxGradient = nextNode->value.maxGradient;

	if (nextNode->parent->value.maxGradient > nextNode->value.maxGradient)
	    break;
	nextNode = nextNode->parent;
    }

    /*fix rb tree after insertion */
    rb_insert_fixupD(root, inserted, NIL);

    return;
}






/*search for a node with the given key */
__device__ TreeNodeD *search_for_nodeD(TreeNodeD * root, double key, TreeNodeD* NIL)
{
    TreeNodeD *curNode = root;

    while (curNode != NIL && compare_doubleD(key, curNode->value.key) != 0) {

	if (compare_doubleD(key, curNode->value.key) == -1) {
	    curNode = curNode->left;
	}
	else {
	    curNode = curNode->right;
	}

    }

    return curNode;
}

/*function used by treeSuccessor */
__device__ TreeNodeD *tree_minimumD(TreeNodeD * x, TreeNodeD* NIL)
{
    while (x->left != NIL)
	x = x->left;

    return x;
}

/*function used by deletion */
__device__ TreeNodeD *tree_successorD(TreeNodeD * x, TreeNodeD* NIL)
{
    if (x->right != NIL)
	return tree_minimumD(x->right, NIL);
    TreeNodeD *y = x->parent;

    while (y != NIL && x == y->right) {
	x = y;
	y = y->parent;
    }
    return y;
}


/*delete the node out of the tree */
__device__ void delete_from_treeD(TreeNodeD ** root, double key, TreeNodeD* NIL)
{
    double tmpMax;
    TreeNodeD *z;
    TreeNodeD *x;
    TreeNodeD *y;
    TreeNodeD *toFix;
    
    z = search_for_nodeD(*root, key, NIL);
    
    if (z == NIL) {
	//fprintf(stderr, "ATTEMPT to delete key=%f failed\n", key);
	//fprintf(stderr, "Node not found. Deletion fails.\n");
	//exit(1);
	return;			/*node to delete is not found */
    }

    /*1-3 */
    if (z->left == NIL || z->right == NIL)
	y = z;
    else
	y = tree_successorD(z, NIL);

    /*4-6 */
    if (y->left != NIL)
	x = y->left;
    else
	x = y->right;

    /*7 */
    x->parent = y->parent;

    /*8-12 */
    if (y->parent == NIL) {
	*root = x;

	toFix = *root;		/*augmentation to be fixed */
    }
    else {
	if (y == y->parent->left)
	    y->parent->left = x;
	else
	    y->parent->right = x;

	toFix = y->parent;	/*augmentation to be fixed */
    }

    /*fix augmentation for removing y */
    TreeNodeD *curNode = y;
    double left, right;

    while (curNode->parent != NIL) {
	if (curNode->parent->value.maxGradient == y->value.gradient) {
	    left = find_max_valueD(curNode->parent->left);
	    right = find_max_valueD(curNode->parent->right);

	    if (left > right)
		curNode->parent->value.maxGradient = left;
	    else
		curNode->parent->value.maxGradient = right;

	    if (curNode->parent->value.gradient >
		curNode->parent->value.maxGradient)
		curNode->parent->value.maxGradient =
		    curNode->parent->value.gradient;
	}
	else {
	    break;
	}
	curNode = curNode->parent;
    }


    /*fix augmentation for x */
    tmpMax =
	toFix->left->value.maxGradient >
	toFix->right->value.maxGradient ? toFix->left->value.
	maxGradient : toFix->right->value.maxGradient;
    if (tmpMax > toFix->value.gradient)
	toFix->value.maxGradient = tmpMax;
    else
	toFix->value.maxGradient = toFix->value.gradient;

    /*13-15 */
    if (y != z) {
	double zGradient = z->value.gradient;

	z->value.key = y->value.key;
	z->value.gradient = y->value.gradient;


	toFix = z;
	/*fix augmentation */
	tmpMax =
	    toFix->left->value.maxGradient >
	    toFix->right->value.maxGradient ? toFix->left->value.
	    maxGradient : toFix->right->value.maxGradient;
	if (tmpMax > toFix->value.gradient)
	    toFix->value.maxGradient = tmpMax;
	else
	    toFix->value.maxGradient = toFix->value.gradient;

	while (z->parent != NIL) {
	    if (z->parent->value.maxGradient == zGradient) {
		if (z->parent->value.gradient != zGradient &&
		    (!(z->parent->left->value.maxGradient == zGradient &&
		       z->parent->right->value.maxGradient == zGradient))) {

		    left = find_max_valueD(z->parent->left);
		    right = find_max_valueD(z->parent->right);

		    if (left > right)
			z->parent->value.maxGradient = left;
		    else
			z->parent->value.maxGradient = right;

		    if (z->parent->value.gradient >
			z->parent->value.maxGradient)
			z->parent->value.maxGradient =
			    z->parent->value.gradient;

		}

	    }
	    else {
		if (z->value.maxGradient > z->parent->value.maxGradient)
		    z->parent->value.maxGradient = z->value.maxGradient;
	    }
	    z = z->parent;
	}

    }

    /*16-17 */
    if (y->color == RB_BLACKD && x != NIL)
	rb_delete_fixupD(root, x, NIL);
	/*18 */
    free(y);


    return;
}

/*find max within the max key */
__device__ float find_max_value_within_keyD(TreeNodeD * root, double maxKey, TreeNodeD* NIL)
{
    TreeNodeD *keyNode = search_for_nodeD(root, maxKey, NIL);

    if (keyNode == NIL) {
	/*fprintf(stderr, "key node not found. error occured!\n");
	   //there is no point in the structure with key < maxKey */
	return SMALLEST_GRADIENTD;	
    }

    double max = find_max_valueD(keyNode->left);
    double tmpMax;

    while (keyNode->parent != NIL) {
	if (keyNode == keyNode->parent->right) {	/*its the right node of its parent; */
	    tmpMax = find_max_valueD(keyNode->parent->left);
	    if (tmpMax > max)
		max = tmpMax;
	    if (keyNode->parent->value.gradient > max)
		max = keyNode->parent->value.gradient;
	}
	keyNode = keyNode->parent;
    }

    return max;
}

/*public:--------------------------------- */
__device__ RBTreeD *create_treeD(TreeValueD tv)
{
	RBTreeD *rbt = (RBTreeD *) malloc(sizeof(RBTreeD));
    init_nil_nodeD(rbt);

    TreeNodeD *root = (TreeNodeD *) malloc(sizeof(TreeNodeD));
    rbt->root = root;
    rbt->root->value = tv;
    rbt->root->left = rbt->NIL;
    rbt->root->right = rbt->NIL;
    rbt->root->parent = rbt->NIL;
    rbt->root->color = RB_BLACKD;

    return rbt;
}

/*LT: not sure if this is correct */
__device__ int is_emptyD(RBTreeD * t, TreeNodeD* NIL)
{
    return (t->root == NIL);
}

__device__ void destroy_sub_treeD(TreeNodeD * node, TreeNodeD* NIL)
{
    if (node == NIL)
	return;
    destroy_sub_treeD(node->left, NIL);
    destroy_sub_treeD(node->right, NIL);

    free(node);
    return;
}



__device__ void delete_treeD(RBTreeD * t)
{
    destroy_sub_treeD(t->root, t->NIL);
    return;
}


__device__ void insert_intoD(RBTreeD * rbt, TreeValueD value)
{
    insert_into_treeD(&(rbt->root), value, rbt->NIL);
    return;
}

__device__ void delete_fromD(RBTreeD * rbt, float key)
{
    delete_from_treeD(&(rbt->root), key, rbt->NIL);
    return;
}

__device__ TreeNodeD *search_for_node_with_keyD(RBTreeD * rbt, float key)
{
    return search_for_nodeD(rbt->root, key, rbt->NIL);
}

/*------------The following is designed for viewshed's algorithm-------*/
__device__ float find_max_gradient_within_keyD(RBTreeD * rbt, float key)
{
    return find_max_value_within_keyD(rbt->root, key, rbt->NIL);
}


__global__ void cudaCalculateEvents(event_t* events, int minX, int maxX, int minY, int maxY, int observerX, int observerY)
{	
	 int col = blockIdx.x * blockDim.x + threadIdx.x;
	 int row = blockIdx.y * blockDim.y + threadIdx.y; 
	 

	 int width = (maxX - minX)+1;
	 int height = (maxY - minY)+1;

	 if(width <= col  || height <= row ) 
	 {
		 return;
	 }

	 int eventIdx = (col + row*width)*3;
	 
		 
	 int x = col + minX;
	 int y = row + minY;
	 if(x == observerX && y == observerY)
	 {
		 events[eventIdx].dist = 0;
		 events[eventIdx].type = ENTERING_EVENT;
		 events[eventIdx].row =observerY; 
		 events[eventIdx].col = observerX;
		 events[eventIdx].angle = 0;

		 events[eventIdx+1].dist = 0;
		 events[eventIdx+1].type = ENTERING_EVENT;
		 events[eventIdx+1].row =observerY; 
		 events[eventIdx+1].col = observerX;
		 events[eventIdx+1].angle = 0;

		 events[eventIdx+2].dist = 0;
		 events[eventIdx+2].type = ENTERING_EVENT;
		 events[eventIdx+2].row = observerY; 
		 events[eventIdx+2].col = observerX;
		 events[eventIdx+2].angle = 0;
		 return;
	 }

	 int deltaY = y - observerY;
	 int deltaX = x - observerX;
	 double slope = double(deltaY)/deltaX;
	 double angle = atan2((double)-deltaY,(double)deltaX);
	 if (angle < 0) angle += 2*5;
	 // calculate enter, center and exit angles depending on cell's quadrant
	double enterOffset[2];
	double exitOffset[2]; 

	if (deltaY<0 && deltaX>0) 
	{ // first quadrant
		enterOffset[0] = +0.5; enterOffset[1] = +0.5; exitOffset[0] = -0.5;	exitOffset[1] = -0.5;
	}
	else if (deltaY<0 && deltaX<0) 
	{ // second quadrant
		enterOffset[0] = -0.5; enterOffset[1] = +0.5; exitOffset[0] = +0.5;	exitOffset[1] = -0.5;
	}
	else if (deltaY>0 && deltaX<0) 
	{ // third quadrant
		enterOffset[0] = -0.5; enterOffset[1] = -0.5; exitOffset[0] = +0.5;	exitOffset[1] = +0.5;
	}
	else if (deltaY>0 && deltaX>0) 
	{ // fourth quadrant
		enterOffset[0] = +0.5; enterOffset[1] = -0.5; exitOffset[0] = -0.5;	exitOffset[1] = +0.5;
	}
	else if (deltaY<0 && deltaX==0) 
	{ // to the north
		enterOffset[0] = +0.5; enterOffset[1] = +0.5; exitOffset[0] = +0.5; exitOffset[1] = -0.5;
	}
	else if (deltaY==0 && deltaX<0) 
	{ // to the west
		enterOffset[0] = -0.5; enterOffset[1] = +0.5; exitOffset[0] = +0.5; exitOffset[1] = +0.5;
	}
	else if (deltaY>0 && deltaX==0) 
	{ // to the south
		enterOffset[0] = -0.5; enterOffset[1] = -0.5; exitOffset[0] = -0.5;	exitOffset[1] = +0.5;
	}
	else if (deltaY==0 && deltaX>0) { // to the east
		enterOffset[0] = +0.5; enterOffset[1] = -0.5; exitOffset[0] = -0.5; exitOffset[1] = -0.5;
	}

	int dy = y - observerY;
	int dx = x - observerX;
	double dist = dx*dx+dy*dy;

	// inserting ENTER event
	double Y = y + enterOffset[0];
	double X = x + enterOffset[1];
	double angleEnter = atan2((double)observerY - Y, (double)X - observerX);
	if (angleEnter < 0) angleEnter += 2*M_PI;

	if(y == observerY && x > observerX)
	{
		angleEnter = -1000;
	}
	events[eventIdx].angle = angleEnter;
	events[eventIdx].dist = dist;
	events[eventIdx].type = ENTERING_EVENT;
	events[eventIdx].row =Y; 
	events[eventIdx].col = X;
	

	// inserting CENTER event
	double angleCenter = atan2((double)observerY - y, (double)x - observerX);
	if (angleCenter < 0) angleCenter += 2*M_PI;
	// just insert it if center is inside this this sector
	events[eventIdx+1].angle = angleCenter;
	events[eventIdx+1].dist = dist;
	events[eventIdx+1].type = CENTER_EVENT;
	events[eventIdx+1].row =events[eventIdx].row ; 
	events[eventIdx+1].col = events[eventIdx].col;

	// inserting EXIT event
	
	Y = y + exitOffset[0]; X = x + exitOffset[1];
	double angleExit = atan2(observerY - Y , X - observerX);
	if (angleExit < 0) angleExit += 2*M_PI;
	events[eventIdx+2].angle = angleExit;
	events[eventIdx+2].dist = dist;
	events[eventIdx+2].type = EXITING_EVENT;
	events[eventIdx+2].row =events[eventIdx].row; 
	events[eventIdx+2].col = events[eventIdx].col;
}


__global__ void testEvent(vs_t* viewshed, int viewshedSize)
{	
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if( index < viewshedSize)
	{
		viewshed[index] = 1;	
	}
}

__global__ void cudaIterateOverEvents(event_t* events, elev_t* elev, vs_t* viewshed, int observerX, int observerY, int eventSize, int radiusX, elev_t observer_elev)
{
	TreeValueD tv;
	tv.gradient = SMALLEST_GRADIENTD;
	tv.key = 0;
	tv.maxGradient = SMALLEST_GRADIENTD;
	RBTreeD* tree = create_treeD(tv);

	
	for(int i = 0; i <  eventSize; i++)
	{
		event_t it = events[i];
		int index = it.row * radiusX + it.col;

		if(it.row == observerY && it.col == observerX)
		{
			continue;
		}

		if (it.type == ENTERING_EVENT) {
			//calculate gradient
			double diff_elev = elev[index] - observer_elev;
			tv.gradient = (diff_elev*diff_elev) / it.dist;
			if (diff_elev < 0) tv.gradient*=-1;

			tv.key = it.dist;
			tv.maxGradient = SMALLEST_GRADIENTD;

			insert_intoD(tree, tv);
		}
		else if (it.type == EXITING_EVENT) {
			delete_fromD(tree,it.dist);
		}
		else { // CENTER_EVENT 
			//calculate gradient
			float diff_elev = elev[index] + /*target_ht*/ - observer_elev;
			float gradient = (diff_elev*diff_elev) / it.dist;
			if (diff_elev <0) gradient*=-1;
			


			float max_grad = find_max_gradient_within_keyD(tree,it.dist);
			if ( max_grad < gradient ) { // the cell is visible!
				viewshed[index] = 1;
			}
			else
			{
				viewshed[index] = 0;
			}
		}
	}
}

__global__ void cudaIterateOverEventsPartialViewshed(event_t* events, elev_t* elev, vs_t* viewshed, int observerX, int observerY, int eventSize, int columnSize, elev_t observer_elev, int minX, int minY,int viewShedColumnSize)
{
	TreeValueD tv;
	tv.gradient = SMALLEST_GRADIENTD;
	tv.key = 0;
	tv.maxGradient = SMALLEST_GRADIENTD;
	RBTreeD* tree = create_treeD(tv);

	
	for(int i = 0; i <  eventSize; i++)
	{
		event_t it = events[i];
		int index = it.row * columnSize + it.col;

		if(it.row == observerY && it.col == observerX)
		{
			continue;
		}

		if (it.type == ENTERING_EVENT) {
			//calculate gradient
			double diff_elev = elev[index] - observer_elev;
			tv.gradient = (diff_elev*diff_elev) / it.dist;
			if (diff_elev < 0) tv.gradient*=-1;

			tv.key = it.dist;
			tv.maxGradient = SMALLEST_GRADIENTD;

			insert_intoD(tree, tv);
		}
		else if (it.type == EXITING_EVENT) {
			delete_fromD(tree,it.dist);
		}
		else { // CENTER_EVENT 
			//calculate gradient
			float diff_elev = elev[index] + /*target_ht*/ - observer_elev;
			float gradient = (diff_elev*diff_elev) / it.dist;
			if (diff_elev <0) gradient*=-1;
			
			int viewshedRow = it.row - minY; 
			int viewshedCol = it.col - minX;
			if(viewshedRow < 0 || viewshedCol < 0)
			{
				continue;
			}
			int viewshedIndex = viewshedRow * viewShedColumnSize + viewshedCol;

			float max_grad = find_max_gradient_within_keyD(tree,it.dist);
			if ( max_grad < gradient ) { // the cell is visible!
				viewshed[viewshedIndex] = 1;
			}
			else
			{
				viewshed[viewshedIndex] = 0;
			}
		}
	}
}

__global__ void cudaR3(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{	
	 int col = blockIdx.x * blockDim.x + threadIdx.x;
	 int row = blockIdx.y * blockDim.y + threadIdx.y; 
	 

	 int width = (maxX - minX)+1;
	 int height = (maxY - minY)+1;

	 if(width <= col  || height <= row ) 
	 {
		 return;
	 }
	 

	

	


	 int x = col + minX;
	 int y = row + minY;

	 int index = (x + y*ncols);

	 if(x == observerX && y == observerY)
	 {
		viewshed[index] = 1;
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
			float dist2 = deltaX*deltaX+deltaY*deltaY;
			int currentIndex = (x1 + y1*ncols);
			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev*diff_elev) / dist2;
			if (diff_elev < 0) gradient*=-1;

			if(y1 == y && x1 == x)
			{
				if(gradient > maxGradient)
				{
					viewshed[index] = 1;
				}
				else
				{
					viewshed[index] = 0;
				}
			}
			else
			{
				if(gradient > maxGradient)
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
			float dist2 = deltaX*deltaX+deltaY*deltaY;

			int currentIndex = (x1 + y1*ncols);

			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev*diff_elev) / dist2;
			if (diff_elev < 0) gradient*=-1;
			if(y1 == y && x1 == x)
			{
				if(gradient > maxGradient)
				{
					viewshed[index] = 1;
				}
				else
				{
					viewshed[index] = 0;
				}
			}
			else
			{
				if(gradient > maxGradient)
				{
					maxGradient = gradient;
				}	
			}
					
		}
	}
}

__global__ void cudaR2(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{

	int idx = blockIdx.x *blockDim.x + threadIdx.x;

	int width = (maxX - minX) + 1;
	int height = (maxY - minY) + 1;

	int totalCell = ((width + height) * 2) - 4;

	if (idx >= totalCell)
	{
		return;
	}

	int x, y;
	if (idx < width)
	{
		x = minX + idx;
		y = minY;
		//return;
	}
	else if (idx >= width && idx < width + height - 1)
	{
		x = maxX;
		y = minY + (idx + 1) - width;
		//return;
	}
	else if (idx >= width + height - 1 && idx <  width + height + width - 2)
	{
		x = maxX - ((idx + 1) - (width + height - 1));
		y = maxY;
	}
	else if (idx >= width + height + width - 2 && idx < totalCell)
	{
		x = minX;
		y = maxY - ((idx + 1) - (width + height + width - 2));
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


			int currentIndex = (x1 + y1*ncols);
			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX*deltaX + deltaY*deltaY;

			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev*diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[currentIndex] = 1;
			}
			else
			{
				viewshed[currentIndex] = 0;
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

			int currentIndex = (x1 + y1*ncols);

			int deltaY = y1 - observerY;
			int deltaX = x1 - observerX;
			float dist2 = deltaX*deltaX + deltaY*deltaY;

			double diff_elev = elev[currentIndex] - observer_elev;
			float gradient = (diff_elev*diff_elev) / dist2;
			if (diff_elev < 0) gradient *= -1;

			if (gradient > maxGradient)
			{
				maxGradient = gradient;
				viewshed[currentIndex] = 1;
			}
			else
			{
				viewshed[currentIndex] = 0;
			}

		}
	}
}



void calculateEventsWrapper(event_t* events, int minX, int maxX, int minY, int maxY, int observerX, int observerY)
{
	int width = (maxX - minX)+1;
	int height = (maxY - minY)+1;
	int eventSize = (width*height*3);

	dim3 dimBlock(32, 32);
	
	dim3 dimGrid((int)std::ceil((float)((float)width/(float)dimBlock.x)),(int)std::ceil((float)((float)height/(float)dimBlock.y)));
	cudaCalculateEvents<<<dimGrid, dimBlock>>>(events,minX,maxX,minY,maxY,observerX,observerY);	
}


void cudaR3Wrapper(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{
	int width = (maxX - minX)+1;
	int height = (maxY - minY)+1;

	dim3 dimBlock(32, 32);
	
	dim3 dimGrid((int)std::ceil((float)((float)width/(float)dimBlock.x)),(int)std::ceil((float)((float)height/(float)dimBlock.y)));
	cudaR3<<<dimGrid, dimBlock>>>(viewshed,elev, observer_elev,minX,maxX,minY,maxY,observerX,observerY, ncols);	
}


void cudaR2Wrapper(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols)
{
	int width = (maxX - minX)+1;
	int height = (maxY - minY)+1;


	int totalCell = ((width + height) * 2) - 4;
	
	
	int size = (int)std::ceil((float)((float)totalCell/(float)1024));
	cudaR2<<<size, 1024>>>(viewshed,elev, observer_elev,minX,maxX,minY,maxY,observerX,observerY, ncols);	
}


void iterateOverEventsWrapper(event_t* events, elev_t* elev, vs_t* viewshed, int observerX, int observerY, int eventSize, int radiusX, elev_t observer_elev)
{	
	//testEvent<<<1,1>>>(viewshed, radiusX);
	cudaIterateOverEvents<<<1, 1>>>(events,elev,viewshed,observerX,observerY,eventSize, radiusX, observer_elev);	
}

void testWrapper(vs_t* viewshed, int viewshedSize)
{
	
	int size = (int)std::ceil((float)((float)viewshedSize/(float)1024));
	
	testEvent<<<size,1024>>>(viewshed, viewshedSize);
}

void iterateOverEventsWrapperPartialViewshed(event_t* events, elev_t* elev, vs_t* viewshed, int observerX, int observerY, int eventSize, int columnSize, elev_t observer_elev, int minX, int minY, int viewShedColumnSize)
{	
	//testEvent<<<1,1>>>(viewshed, radiusX);
	cudaIterateOverEventsPartialViewshed<<<1, 1>>>(events,elev,viewshed,observerX,observerY,eventSize, columnSize, observer_elev, minX, minY, viewShedColumnSize);	
}