#include <stdio.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdbool.h>
using namespace cv;
using namespace std;

#define DISPLAY 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      
	  if(abort)
	  	exit(code);
   }
}


typedef struct
{
	float x, y, w, h, s;
}box;

__device__
float IOUcalc(box b1, box b2)
{
	//Get Union of boxes
	float tlx_u = fmin(b1.x, b2.x);
	float tly_u = fmin(b1.y, b2.y);

	float brx_u = fmax(b1.x + b1.w, b2.x + b2.w);
	float bry_u = fmax(b1.y + b1.h, b2.y + b2.h);

	//Get Intersection of boxes
	float tlx_i = fmax(b1.x, b2.x);
	float tly_i = fmax(b1.y, b2.y);

	float brx_i = fmin(b1.x + b1.w, b2.x + b2.w);
	float bry_i = fmin(b1.y + b1.h, b2.y + b2.h);

	float w_u = brx_u - tlx_u;
	float h_u = bry_u - tly_u;

	float w_i = brx_i - tlx_i;
	float h_i = bry_i - tly_i;

	float inter = (w_i * h_i) / (w_u * h_u);

	return inter;
}

__global__
void NMS_GPU(box *d_b, bool *d_res, const float thd, const float theta)
{
	int target = blockIdx.x;
	int current = threadIdx.x;
	int neighbor = 0;
	int n_theta = 0.3f;

	if(d_b[target].s > d_b[current].s)
	{
		float iou = IOUcalc(d_b[target], d_b[current]);
		if(iou > theta)
			d_res[current] = false;
	}
}


int main()
{
	int count = 6;
	Mat input = imread("./0.jpg",1);
#if DISPLAY
	imshow("Input", input);
	waitKey(0);
#endif	
	bool *h_res =(bool *)malloc(sizeof(bool)*count);
	
	for(int i = 0; i < count; i++)
		h_res[i] = true;
	
	box b[count];
	b[5].x = 155; b[5].y = 30; b[5].w = 70; b[5].h = 138; b[5].s = 0.5355;
	b[4].x = 150; b[4].y = 25; b[4].w = 74; b[4].h = 148; b[4].s = 0.2355;
	b[3].x = 11; b[3].y = 6; b[3].w = 74; b[3].h = 148; b[3].s = 0.42355;
	b[2].x = 12; b[2].y = 14; b[2].w = 70; b[2].h = 141; b[2].s = 0.60434;
	b[1].x = 16; b[1].y = 12; b[1].w = 64; b[1].h = 128; b[1].s = 0.79062;
	b[0].x = 11; b[0].y = 6; b[0].w = 74; b[0].h = 148; b[0].s = 0.11855;
	
#if DISPLAY
	Mat temp = input.clone();
	for(int i = 0; i < count ; i++)
		rectangle(temp, Point(b[i].x,b[i].y), Point(b[i].x + b[i].w,b[i].y + b[i].h), Scalar(0,255,0), 1, 8, 0);
	imshow("Temp", temp);
	waitKey(0);
#endif

	box *d_b;
	bool *d_res;
	
	gpuErrchk(cudaMalloc((void**)&d_res, count*sizeof(bool)));
	gpuErrchk(cudaMemcpy(d_res, h_res,sizeof(bool) * count, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&d_b, sizeof(box) * count));
	gpuErrchk(cudaMemcpy(d_b, b, sizeof(box) * count, cudaMemcpyHostToDevice));
		
	NMS_GPU <<< count, count >>> (d_b, d_res, 0.8f, 0.3f);
	
	cudaThreadSynchronize();
	
	gpuErrchk(cudaMemcpy(h_res, d_res, sizeof(bool) * count, cudaMemcpyDeviceToHost));
	
	for(int i = 0; i < count ; i++)
	{
		printf("Results= %d--%d\n",i ,h_res[i]);
		if(h_res[i] == true)
		{
			rectangle(input, Point(b[i].x, b[i].y), Point(b[i].x + b[i].w, b[i].y + b[i].h), Scalar(255,0,0), 1, 8, 0);
			fflush(stdout);
		}
	}

#if DISPLAY
	imshow("Output",input);
	waitKey(0);
#endif
	return 0;
}
