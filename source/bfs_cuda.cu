#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <unistd.h>
#include <sys/time.h>
#include <cstring>
#include <cmath>
#define LOCAL_FRONTIER_CAPACITY 100
using namespace std;
extern "C" {
	void gpuBFS(
		int my_rank,
		unsigned int* srcPtrs,
		unsigned int lenSrcPtrs,
		unsigned int* d_srcPtrs,
		unsigned int* dst,
		unsigned int lenDst,
		unsigned int* d_dst,
		unsigned int* level,
		unsigned int* prevFrontier,
		unsigned int* currFrontier,
		unsigned int numPrevFrontier,
		unsigned int* numCurrFrontier,
		unsigned int** d_level,
		unsigned int** d_prevFrontier,
		unsigned int** d_currFrontier,
		unsigned int** d_numCurrFrontier,
		unsigned int work,
		unsigned int currLevel
	);
	void copyCSRToGPU(
		unsigned int *srcPtrs, unsigned int lenSrcPtrs, unsigned int **d_srcPtrs,
		unsigned int *dst, unsigned int lenDst, unsigned int **d_dst,
		unsigned int **d_level,unsigned int **d_prevFrontier,unsigned int **d_currFrontier,unsigned int **d_numCurrFrontier);
	void freeDeviceMemory(unsigned int *d_srcPtrs, unsigned int *d_dst,unsigned int *d_level,unsigned int *d_prevFrontier,unsigned int *d_currFrontier,unsigned int *d_numCurrFrontier);
	void updateLevel(
		int my_rank,
		unsigned int *d_level,
		unsigned int *d_currFrontier,
		unsigned int *level,
		unsigned int *currFrontier,
		unsigned int numVertices,
		unsigned int numCurrFrontier,
		unsigned int currLevel
	);
	void copyLevelToGPU(
		unsigned int *level,
		unsigned int** d_level,
		unsigned int numVertices
	);
	void copyLevelToHost(
		unsigned int *level,
		unsigned int** d_level,
		unsigned int numVertices
	);
}

__global__ void bfs_kernel_shared_memory(
	unsigned int* srcPtrs,
	unsigned int* dst,
	unsigned int* level,
	unsigned int* prevFrontier,
	unsigned int* currFrontier,
	unsigned int numPrevFrontier,
	unsigned int* numCurrFrontier,
	unsigned int currLevel,
	unsigned int numVertices)
{


	__shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
	__shared__ unsigned int numCurrFrontier_s;
	if (threadIdx.x == 0) {
	  numCurrFrontier_s = 0;
	}
	__syncthreads();
  
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numPrevFrontier)
	{
		unsigned int vertex = prevFrontier[i];
		for (unsigned int edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; edge++)
		{
			unsigned int neighbor = dst[edge];
			if (atomicCAS(&level[neighbor], UINT32_MAX, currLevel) == UINT32_MAX)
			{
				unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
				if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY)
				{
					currFrontier_s[currFrontierIdx_s] = neighbor;
				}
				else
				{
					numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
					unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
					currFrontier[currFrontierIdx] = neighbor;
				}
			}
		}
	}
	__syncthreads();
  
	// Allocate in global frontier
	__shared__ unsigned int currFrontierStartIndex;
	if (threadIdx.x == 0) {
	  currFrontierStartIndex = atomicAdd(numCurrFrontier, numCurrFrontier_s);
	}
	__syncthreads();
  
	// Commit to global frontier
	for (unsigned int currFrontierIdx_s = threadIdx.x;currFrontierIdx_s < numCurrFrontier_s;currFrontierIdx_s += blockDim.x) {
	  unsigned int currFrontierIdx = currFrontierStartIndex + currFrontierIdx_s;
	  currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
	}
}

__global__ void bfs_kernel_global_memory(
	unsigned int* srcPtrs,
	unsigned int* dst,
	unsigned int* level,
	unsigned int* prevFrontier,
	unsigned int* currFrontier,
	unsigned int numPrevFrontier,
	unsigned int* numCurrFrontier,
	unsigned int currLevel,
	unsigned int numVertices)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numPrevFrontier)
	{
		unsigned int vertex = prevFrontier[i];
		for (unsigned int edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; edge++)
		{
			unsigned int neighbor = dst[edge];
			if (atomicCAS(&level[neighbor], UINT32_MAX, currLevel) == UINT32_MAX)
			{
				unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
				currFrontier[currFrontierIdx] = neighbor;
			}
		}
	}
}

void h_cpuBFS(
	unsigned int* srcPtrs,
	unsigned int* dst,
	unsigned int* level,
	unsigned int* prevFrontier,
	unsigned int* currFrontier,
	unsigned int numPrevFrontier,
	unsigned int* numCurrFrontier,
	unsigned int currLevel)
{
	for(unsigned int i=0;i<numPrevFrontier;i++){
		unsigned int vertex = prevFrontier[i];
		/*if(prevFrontier[0]==8 and numPrevFrontier==1){
			cout<<"BFS for vertex 3"<<endl;
			cout<<"vertex "<<vertex<<" next vertex "<<vertex+1<<endl;
			cout<<"initial edge "<<srcPtrs[vertex]<<" final edge "<< srcPtrs[vertex + 1]<<endl;
			for(int k=0;k<10;k++){
				cout<<srcPtrs[k]<<"--";
			}
			cout<<endl;
		}*/
		for(unsigned int edge= srcPtrs[vertex];edge < srcPtrs[vertex + 1]; edge++){
			/*if(prevFrontier[0]==8 and numPrevFrontier==1){
				cout<<"edge requested"<<edge<<endl;
			}*/
			unsigned int neighbor = dst[edge];
			if(level[neighbor]==UINT32_MAX){
				level[neighbor]=currLevel;
				currFrontier[*numCurrFrontier]=neighbor;
				*numCurrFrontier=*numCurrFrontier+1;
			}
		}
	}
}

void printDeviceProperties(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	printf("prop.name=%s\n", prop.name);
	printf("prop.multiProcessorCount=%d\n", prop.multiProcessorCount);
	printf("prop.major=%d minor=%d\n", prop.major, prop.minor);
	printf("prop.maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim.x=%d maxThreadsDim.y=%d maxThreadsDim.z=%d\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
	printf("prop.maxGridSize.x=%d maxGridSize.y=%d maxGridSize.z=%d\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	printf("prop.maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
	printf("prop.totalGlobalMem=%lu\n", prop.totalGlobalMem);
	printf("prop.regsPerBlock=%d\n", prop.regsPerBlock);
	printf("\n");
}

void copyCSRToGPU(
	unsigned int *srcPtrs, unsigned int lenSrcPtrs, unsigned int **d_srcPtrs,
	unsigned int *dst, unsigned int lenDst, unsigned int **d_dst,
	unsigned int **d_level,unsigned int **d_prevFrontier,unsigned int **d_currFrontier,unsigned int **d_numCurrFrontier)
{	
	printDeviceProperties();
	cudaMalloc((void **)d_srcPtrs, lenSrcPtrs * sizeof(unsigned int));
	cudaError_t err = cudaGetLastError();
	cudaMemcpy(*d_srcPtrs, srcPtrs, lenSrcPtrs * sizeof(unsigned int), cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	cudaMalloc((void **)d_dst, lenDst * sizeof(unsigned int));
	err = cudaGetLastError();
	cudaMemcpy(*d_dst, dst, lenDst * sizeof(unsigned int), cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	cudaMalloc(d_level, sizeof(unsigned int)*(lenSrcPtrs-1));
	err = cudaGetLastError();
	err = cudaGetLastError();
	cudaMalloc(d_prevFrontier, sizeof(unsigned int)*(lenSrcPtrs-1));
	err = cudaGetLastError();
	cudaMalloc(d_currFrontier, sizeof(unsigned int)*(lenSrcPtrs-1));
	err = cudaGetLastError();
	cudaMalloc(d_numCurrFrontier, sizeof(unsigned int));
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		cout<<"CUDA creating memory error : " << cudaGetErrorString(err) <<endl;
	}

}
void copyLevelToGPU(
	unsigned int *level,
	unsigned int** d_level,
	unsigned int numVertices
)
{
	cudaError_t err = cudaGetLastError();
	cudaMemcpy(*d_level, level, sizeof(unsigned int)*numVertices, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout<< "CUDA gpuBFS Error Copying Level to gpu " << cudaGetErrorString(err) <<endl;
	}
}

void copyLevelToHost(
	unsigned int *level,
	unsigned int** d_level,
	unsigned int numVertices
){
	cudaError_t err = cudaGetLastError();
	cudaMemcpy(level,*d_level,sizeof(unsigned int)*numVertices, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout<<"For my rank "<< "CUDA gpuBFS Error Copying Level: to host " << cudaGetErrorString(err) <<endl;
	}
}

// Function to free device memory
void freeDeviceMemory(unsigned int *d_srcPtrs, unsigned int *d_dst,unsigned int *d_level,unsigned int *d_prevFrontier,unsigned int *d_currFrontier,unsigned int *d_numCurrFrontier)
{
	if (d_srcPtrs != nullptr)
	{
		cudaFree(d_srcPtrs);
	}
	if (d_dst != nullptr)
	{
		cudaFree(d_dst);
	}
	if (d_level != nullptr)
	{
		cudaFree(d_level);
	}
	if (d_prevFrontier != nullptr)
	{
		cudaFree(d_prevFrontier);
	}
	if (d_currFrontier != nullptr)
	{
		cudaFree(d_currFrontier);
	}
	if (d_numCurrFrontier != nullptr)
	{
		cudaFree(d_numCurrFrontier);
	}
}

__global__ void updateLevelKernel(unsigned int *d_level,unsigned int *d_currFrontier,unsigned int numCurrFrontier,unsigned int currLevel){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	/*
	if (i < numCurrFrontier) {
		for(int j=0;j<=64;j++){
			if(i+j<numCurrFrontier){
				d_level[d_currFrontier[i+j]]=currLevel;
			}
		}
		
	}*/
	if (i < numCurrFrontier){
		d_level[d_currFrontier[i]]=currLevel;
	}
}

void updateLevel(
	int my_rank,
	unsigned int *d_level,
	unsigned int *d_currFrontier,
	unsigned int *level,
	unsigned int *currFrontier,
	unsigned int numVertices,
	unsigned int numCurrFrontier,
	unsigned int currLevel
){
	cudaMemcpy(d_currFrontier, currFrontier, sizeof(unsigned int)*numCurrFrontier, cudaMemcpyHostToDevice);
	dim3 gridSize(ceil(numCurrFrontier/(32))+1);
	dim3 blockSize(32);
	updateLevelKernel<<<gridSize,blockSize>>>(d_level,d_currFrontier,numCurrFrontier,currLevel);
	cudaMemcpy(level, d_level, sizeof(unsigned int)*numVertices,cudaMemcpyDeviceToHost);
	fflush(stdout);
}

bool compareCPU(unsigned int* d_currFrontier, unsigned int* h_currentFrontier, unsigned int d_numCurrFrontier, unsigned int h_numCurrFrontier){
	if(h_numCurrFrontier!=d_numCurrFrontier){
		return false;
	}
	else{
		for(unsigned int k=0;k<h_numCurrFrontier;k++){
			if(d_currFrontier[k]!=h_currentFrontier[k]){
				return false;
			}
		}
	}
	return true;

}
void gpuBFS(
	int my_rank,
	unsigned int* srcPtrs,
	unsigned int lenSrcPtrs,
	unsigned int* d_srcPtrs,
	unsigned int* dst,
	unsigned int lenDst,
	unsigned int* d_dst,
	unsigned int* level,
	unsigned int* prevFrontier,
	unsigned int* currFrontier,
	unsigned int numPrevFrontier,
	unsigned int* numCurrFrontier,
	unsigned int** d_level,
	unsigned int** d_prevFrontier,
	unsigned int** d_currFrontier,
	unsigned int** d_numCurrFrontier,
	unsigned int work,
	unsigned int currLevel)
{
	cudaError_t err = cudaGetLastError();
	cudaMemcpy(*d_prevFrontier, prevFrontier+(my_rank*work), sizeof(unsigned int)*numPrevFrontier, cudaMemcpyHostToDevice);
	cudaMemcpy(*d_numCurrFrontier, numCurrFrontier, sizeof(unsigned int)*1, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout<<"For my rank "<< "CUDA initial memcpy error : " << cudaGetErrorString(err) <<endl;
	}
	dim3 gridSize(ceil(numPrevFrontier/32)+1);
	dim3 blockSize(32);

	cout<<"At level "<<currLevel<<" Grid size "<<gridSize.x<<" gpu numPrevFrontier "<<numPrevFrontier<<endl;
	//bfs_kernel_global_memory<<<gridSize, blockSize>>>(d_srcPtrs,d_dst,*d_level,*d_prevFrontier,*d_currFrontier,numPrevFrontier,*d_numCurrFrontier,currLevel,(lenSrcPtrs-1));
	bfs_kernel_shared_memory<<<gridSize, blockSize>>>(d_srcPtrs,d_dst,*d_level,*d_prevFrontier,*d_currFrontier,numPrevFrontier,*d_numCurrFrontier,currLevel,(lenSrcPtrs-1));
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		cout<<"For my rank "<< "CUDA bfs kernel error : " << cudaGetErrorString(err) <<endl;
	}
	
	cudaMemcpy(numCurrFrontier,*d_numCurrFrontier,sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(currFrontier,*d_currFrontier,sizeof(unsigned int)*(*numCurrFrontier), cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		cout<<"For my rank "<< "CUDA memcpy error : " << cudaGetErrorString(err) <<endl;
	}
	cout<<"For my rank "<<my_rank<<" At level "<<currLevel<<" gpu Current frontier "<<*numCurrFrontier<<endl;
	if (err != cudaSuccess) {
		cout<<"For my rank "<< "CUDA gpuBFS Error : " << cudaGetErrorString(err) <<endl;
	}
	fflush(stdout);


}