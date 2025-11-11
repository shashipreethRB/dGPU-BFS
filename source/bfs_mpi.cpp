#include <mpi.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <numeric>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <iterator>
#include <sys/time.h>
#define ROOT 0
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

void cpuBFS(
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
		for(unsigned int edge= srcPtrs[vertex];edge < srcPtrs[vertex + 1]; edge++){
			unsigned int neighbor = dst[edge];
			if(level[neighbor]==UINT32_MAX){
				level[neighbor]=currLevel;
				currFrontier[*numCurrFrontier]=neighbor;
				*numCurrFrontier=*numCurrFrontier+1;
			}
		}
	}
}



void frontierParallel(
	int my_rank,
	int nprocs,
	unsigned int lenSrcPtrs,
	unsigned int *srcPtrs,
	unsigned int *d_srcPtrs,
	unsigned int lenDst,
	unsigned int *dst,
	unsigned int *d_dst,
	unsigned int *level,
	unsigned int vertex,
	unsigned int **d_level,
	unsigned int **d_prevFrontier,
	unsigned int **d_currFrontier,
	unsigned int **d_numCurrFrontier
){
	MPI_Comm world = MPI_COMM_WORLD;
	unsigned int numVertices = lenSrcPtrs-1;
	unsigned int numPrevFrontier=1, numCurrFrontier=0, currLevel=1, myNumCurrFrontier=0;
	unsigned int *prevFrontier;
	unsigned int *currFrontier = (unsigned int*) malloc(sizeof(unsigned int)*numVertices);
	unsigned int *myCurrFrontier = (unsigned int*) malloc(sizeof(unsigned int)*numVertices);
	unsigned int *myPrevFrontier = (unsigned int*) malloc(sizeof(unsigned int)*numVertices);
	unsigned int *allNumCurrFrontier = (unsigned int*) malloc(sizeof(unsigned int)*nprocs);
	int *recvCounts = (int*) malloc(sizeof(int)*nprocs);
	int *displacements = (int*) malloc(sizeof(int)*nprocs);
	if(my_rank==0){
		for(unsigned int h=0;h<numVertices;h++) level[h]=UINT32_MAX;
		prevFrontier = (unsigned int*) malloc(sizeof(unsigned int)*numVertices);
		prevFrontier[0]=vertex;
		level[vertex]=0;
		cpuBFS(srcPtrs,dst,level,prevFrontier,currFrontier,numPrevFrontier,&numCurrFrontier,currLevel);
		while(numCurrFrontier<1000){
			//Local BFS as here only one vertex is the frontier
			
			currLevel++;
			numPrevFrontier = numCurrFrontier;
			for(unsigned int pf=0;pf<numPrevFrontier;pf++){
				prevFrontier[pf]=currFrontier[pf];
			}
			numCurrFrontier=0;
			cpuBFS(srcPtrs,dst,level,prevFrontier,currFrontier,numPrevFrontier,&numCurrFrontier,currLevel);

		}
	}
	//Initial Scatter from ROOT
	MPI_Barrier(world);
	MPI_Bcast(&numCurrFrontier,1,MPI_UINT32_T,ROOT,world);
	MPI_Bcast(&currLevel,1,MPI_INT,ROOT,world);
	MPI_Barrier(world);
	if(numCurrFrontier>0){
		currLevel+=1;
		MPI_Barrier(world);
		MPI_Bcast(level,numVertices,MPI_UINT32_T,ROOT,world);
		MPI_Bcast(currFrontier,numCurrFrontier,MPI_UINT32_T,ROOT,world);
		MPI_Barrier(world);
		int my_work = numCurrFrontier / nprocs;
		int work = my_work;
		if(my_rank==nprocs-1){
			my_work=my_work+(numCurrFrontier%nprocs);
		};
		cout<<"Initial Scatter For Rank "<< my_rank<<" numCurrFrontier "<<numCurrFrontier<<" numVertices "<<numVertices<<" my work "<<my_work<<endl;
		cpuBFS(srcPtrs,dst,level,currFrontier+(my_rank*work),myCurrFrontier,my_work,&myNumCurrFrontier,currLevel);
		//gpuBFS(my_rank,srcPtrs,lenSrcPtrs,d_srcPtrs,dst,lenDst,d_dst,level,currFrontier,myCurrFrontier,my_work,&myNumCurrFrontier,d_level,d_prevFrontier,d_currFrontier,d_numCurrFrontier,work,currLevel);
		cout<<"For Rank "<< my_rank<<" Level 2ndary BFS has ended"<<endl;
		cout<<"For Rank "<< my_rank<<" No of frontiers discovered"<<myNumCurrFrontier<<endl;
		MPI_Barrier(world);
		MPI_Allgather(&myNumCurrFrontier,1,MPI_UINT32_T,allNumCurrFrontier,1,MPI_UINT32_T,world);
		MPI_Barrier(world);
		numCurrFrontier = accumulate(allNumCurrFrontier, allNumCurrFrontier + nprocs, 0);
		cout<<"For Rank "<< my_rank<<" Total No of frontiers discovered"<<numCurrFrontier<<endl;
		for(int pv=0;pv<nprocs;pv++){
			recvCounts[pv]=allNumCurrFrontier[pv];
			if(pv==0){
				displacements[pv]=0;
			}
			else{
				displacements[pv]=displacements[pv-1]+recvCounts[pv-1];
			}
		}
		MPI_Barrier(world);
		MPI_Allgatherv(myCurrFrontier,myNumCurrFrontier,MPI_UINT32_T,currFrontier,recvCounts,displacements,MPI_UINT32_T,world);
		MPI_Barrier(world);
		/*for(int uv=0;uv<numCurrFrontier;uv++){
			mySet.insert(currFrontier[uv]);
		}
		int jt=0;
		for(auto it = mySet.begin(); it != mySet.end(); it++,jt++){
			currFrontier[jt]=*it;
		}
		numCurrFrontier=mySet.size();
		mySet.clear();
		*/

		//cout<<"For Rank "<< my_rank<<" All frontiers gathered"<<" unique count "<<numCurrFrontier<<endl;
		//if(numCurrFrontier<10000){
			for(unsigned int v=0;v<numCurrFrontier;v++){
				level[currFrontier[v]]=currLevel;
			}
		//}
		//else{
		//	updateLevel(my_rank,*d_level,*d_currFrontier,level,currFrontier,numVertices,numCurrFrontier,currLevel);
		//}
		cout<<"Initial Scatter from root complete"<<endl;
	}
	copyLevelToGPU(level,d_level,numVertices);
	MPI_Barrier(world);
	while (numCurrFrontier>0 && numCurrFrontier>1000)
	{
		currLevel+=1;
		myNumCurrFrontier=0;
		int my_work = numCurrFrontier / nprocs;
		int work = my_work;
		/*for(unsigned int w=0;w<my_work;w++){
			myPrevFrontier[w]=currFrontier[w+(my_rank*my_work)];
		}*/
		if(my_rank==nprocs-1){
			/*for(unsigned int w=0;w<(numCurrFrontier%nprocs);w++){
				myPrevFrontier[my_work+w]=currFrontier[w+((my_rank+1)*my_work)];
			}*/
			my_work=my_work+(numCurrFrontier%nprocs);
		};
		cout<<"Loop BFS For Rank "<< my_rank<<" Level "<<currLevel<<" numCurrFrontier "<<numCurrFrontier<<" numVertices "<<numVertices<<" my work "<<my_work<<endl;
		//cpuBFS(srcPtrs,dst,level,myPrevFrontier,myCurrFrontier,my_work,&myNumCurrFrontier,currLevel);
		gpuBFS(my_rank,srcPtrs,lenSrcPtrs,d_srcPtrs,dst,lenDst,d_dst,level,currFrontier,myCurrFrontier,my_work,&myNumCurrFrontier,d_level,d_prevFrontier,d_currFrontier,d_numCurrFrontier,work,currLevel);
		cout<<"Loop BFS For Ended Rank "<< my_rank<<" Level "<<currLevel<<endl;
		//MPI_Barrier(world);
		MPI_Allgather(&myNumCurrFrontier,1,MPI_UINT32_T,allNumCurrFrontier,1,MPI_UINT32_T,world);
		numCurrFrontier = accumulate(allNumCurrFrontier, allNumCurrFrontier + nprocs, 0);
		//MPI_Barrier(world);
		cout<<"For Loop Rank "<< my_rank<<" Level "<<currLevel<<" Total No of frontiers discovered "<<numCurrFrontier<<endl;
		for(int pv=0;pv<nprocs;pv++){
			recvCounts[pv]=allNumCurrFrontier[pv];
			if(pv==0){
				displacements[pv]=0;
			}
			else{
				displacements[pv]=displacements[pv-1]+recvCounts[pv-1];
			}
		}
		MPI_Allgatherv(myCurrFrontier,myNumCurrFrontier,MPI_UINT32_T,currFrontier,recvCounts,displacements,MPI_UINT32_T,world);
		//MPI_Barrier(world);
		/*for(int uv=0;uv<numCurrFrontier;uv++){
			mySet.insert(currFrontier[uv]);
		}
		int jt=0;
		for(auto it = mySet.begin(); it != mySet.end(); it++,jt++){
			currFrontier[jt]=*it;
		}
		numCurrFrontier=mySet.size();
		mySet.clear();*/
		cout<<"For Rank "<< my_rank<<" All frontiers gathered"<<" New unique count "<<numCurrFrontier<<endl;
		cout<<endl;
		//if(numCurrFrontier<10000){
		//	for(unsigned int v=0;v<numCurrFrontier;v++){
		//		level[currFrontier[v]]=currLevel;
		//	}
		//}
		//else{
			updateLevel(my_rank,*d_level,*d_currFrontier,level,currFrontier,numVertices,numCurrFrontier,currLevel);
		//}
	}
	MPI_Barrier(world);
	if(numCurrFrontier>0){
		MPI_Barrier(world);
		if(my_rank==0){
			copyLevelToHost(level,d_level,numVertices);
			cout<<"Entered last loop to complete BFS"<<endl;
			while (numCurrFrontier>0)
			{
				currLevel+=1;
				numPrevFrontier = numCurrFrontier;
				for(unsigned int pf=0;pf<numPrevFrontier;pf++){
					prevFrontier[pf]=currFrontier[pf];
				}
				numCurrFrontier=0;
				cpuBFS(srcPtrs,dst,level,prevFrontier,currFrontier,numPrevFrontier,&numCurrFrontier,currLevel);
			}
		}
	}
	
	
}

void loadCSR(string filePath, unsigned int** srcPtrs, unsigned int* lenSrcPtrs, unsigned int** dst, unsigned int* lenDst){
	ifstream file(filePath);
	if (!file.is_open()) {
		cerr << "Error opening file!" << endl;
		return;
	}

	string line;
	if (getline(file, line)) {
		vector<int> numbers;
		stringstream ss(line);
		string token;

		while (getline(ss, token, ',')) {
			try {
				numbers.push_back(stoi(token));
			} catch (const invalid_argument& e) {
				cerr << "Invalid number found: " << token << endl;
			}
		}

		*lenSrcPtrs = numbers.size();
		*srcPtrs = (unsigned int*)malloc(sizeof(unsigned int) * (*lenSrcPtrs));
		for (size_t i = 0; i < numbers.size(); ++i) {
			(*srcPtrs)[i] = numbers[i];
		}
	}

	if (getline(file, line)) {
		vector<int> numbers2;
		stringstream ss(line);
		string token;
		while (getline(ss, token, ',')) {
			try {
				numbers2.push_back(stoi(token));
			} catch (const invalid_argument& e) {
				cerr << "Invalid number found: " << token << endl;
			}
		}

		*lenDst = numbers2.size();
		*dst = (unsigned int*)malloc(sizeof(unsigned int) * (*lenDst));
		for (size_t i = 0; i < numbers2.size(); ++i) {
			(*dst)[i] = numbers2[i];
		}
	}
}

int main(int argc, char *argv[]){
	long mpi_start, mpi_end, mpi_elapsed;
	struct timeval timecheck;
	int nprocs, my_rank;
	unsigned int lenSrcPtrs, lenDst;
	unsigned int *srcPtrs, *dst, *d_srcPtrs, *d_dst, *d_level, *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
	MPI_Comm world = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	string filepath;
	unsigned int vertex=0;
	if(argc>1){
		filepath=argv[1];
		vertex = atoi(argv[2]);
	}
	else{
		filepath="/home/mpiuser/Downloads/csr_roadNet.csv";
	}
	if(my_rank==0){
		//load csr format of graph
		/*unsigned int lSrcPtrs[10] = {0,2,4,7,9,11,12,13,15,15};
		unsigned int  lDst[15] = {1,2,3,4,5,6,7,4,8,5,8,6,8,0,6};
		lenSrcPtrs = sizeof(lSrcPtrs)/sizeof(lSrcPtrs[0]);
		lenDst = sizeof(lDst)/sizeof(lDst[0]);
		srcPtrs = (unsigned int *) malloc(sizeof(unsigned int)*lenSrcPtrs);
		dst = (unsigned int *) malloc(sizeof(unsigned int)*lenDst);
		memcpy(srcPtrs,lSrcPtrs,sizeof(lSrcPtrs));
		memcpy(dst,lDst,sizeof(lDst));*/
		loadCSR(filepath, &srcPtrs, &lenSrcPtrs, &dst, &lenDst);
	}
	//Transmist CSR to all nodes
	MPI_Barrier(world);
	MPI_Bcast(&lenSrcPtrs, 1, MPI_UINT32_T, ROOT,world);
	MPI_Barrier(world);
	MPI_Bcast(&lenDst, 1, MPI_UINT32_T, ROOT,world);
	MPI_Barrier(world);
	cout<<"For Rank "<<my_rank<<" the length of srcPtrs: "<<lenSrcPtrs<<" dst: "<<lenDst<<endl;
	if(my_rank!=0){
		srcPtrs = (unsigned int *) malloc(sizeof(unsigned int)*lenSrcPtrs);
		dst = (unsigned int *) malloc(sizeof(unsigned int)*lenDst);
	}
	cout<<"For Rank "<<my_rank<<" srcPrts and dst arrays are created"<<endl;
	MPI_Barrier(world);
	MPI_Bcast(srcPtrs, lenSrcPtrs, MPI_UINT32_T, ROOT,world);
	MPI_Barrier(world);
	MPI_Bcast(dst, lenDst, MPI_UINT32_T, ROOT,world);
	MPI_Barrier(world);
	copyCSRToGPU(srcPtrs, lenSrcPtrs,&d_srcPtrs,dst,lenDst,&d_dst,&d_level,&d_prevFrontier,&d_currFrontier,&d_numCurrFrontier);
	//unsigned int vertex=1000;
	unsigned int* level	 = (unsigned int*) malloc(sizeof(unsigned int)*lenSrcPtrs);
	cout<<"For Rank "<<my_rank<<"frontier search cstarted"<<endl;
	MPI_Barrier(world);
	gettimeofday(&timecheck, NULL);
	mpi_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
	frontierParallel(
		my_rank,nprocs,
		lenSrcPtrs,srcPtrs,d_srcPtrs,
		lenDst,dst,d_dst,
		level,
		vertex,&d_level,&d_prevFrontier,&d_currFrontier,&d_numCurrFrontier
	);
	gettimeofday(&timecheck, NULL);
	mpi_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
	mpi_elapsed = mpi_end - mpi_start;
	printf("mpi time: rank=%d: %d procs: %ld msecs\n",my_rank, nprocs, mpi_elapsed);
	MPI_Barrier(world);
	cout<<"For Rank "<<my_rank<<"frontier search completed"<<endl;
	freeDeviceMemory(d_srcPtrs,d_dst,d_level,d_prevFrontier,d_currFrontier,d_numCurrFrontier);
	MPI_Finalize();
	int ccount=0;
	int maxcount=0;
	if(my_rank==0){
		for(unsigned int h=0;h<lenSrcPtrs-1;h++){
			if(level[h]!=UINT32_MAX){
				ccount++;
				if(level[h]>maxcount){
					maxcount=level[h];
				}
			}

		}
		cout<<" Number of reachable vertices "<<ccount<<"Max level"<<maxcount<<endl;
		//local bfs
		gettimeofday(&timecheck, NULL);
		mpi_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
		unsigned int* l_level = (unsigned int*) malloc(sizeof(unsigned int)*lenSrcPtrs);
		unsigned int *myCurrFrontier = (unsigned int*) malloc(sizeof(unsigned int)*lenSrcPtrs);
		unsigned int *myPrevFrontier = (unsigned int*) malloc(sizeof(unsigned int)*lenSrcPtrs);
		unsigned int myNumPrevFrontier, myNumCurrFrontier=0, l_currLevel=1;
		myNumPrevFrontier=1;
		myPrevFrontier[0]=vertex;
		for(unsigned int h=0;h<lenSrcPtrs;h++) l_level[h]=UINT32_MAX;
		l_level[0]=0;
		cpuBFS(srcPtrs,dst,l_level,myPrevFrontier,myCurrFrontier,myNumPrevFrontier,&myNumCurrFrontier,l_currLevel);
		while(myNumCurrFrontier>0){
			//Local BFS as here only one vertex is the frontier
			
			l_currLevel+=1;
			myNumPrevFrontier = myNumCurrFrontier;
			for(unsigned int pf=0;pf<myNumPrevFrontier;pf++){
				myPrevFrontier[pf]=myCurrFrontier[pf];
			}
			myNumCurrFrontier=0;
			cpuBFS(srcPtrs,dst,l_level,myPrevFrontier,myCurrFrontier,myNumPrevFrontier,&myNumCurrFrontier,l_currLevel);

		}
		ccount=0;
		maxcount=0;
		for(unsigned int h=0;h<lenSrcPtrs-1;h++){
			if(l_level[h]!=UINT32_MAX){
				ccount++;
				if(l_level[h]>maxcount){
					maxcount=l_level[h];
				}
			}

		}
		cout<<" Number of reachable vertices by CPU "<<ccount<<"Max level"<<maxcount<<endl;
		gettimeofday(&timecheck, NULL);
		mpi_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
		mpi_elapsed = mpi_end - mpi_start;
		printf("cpu time: rank=%d: %d procs: %ld msecs\n",my_rank, nprocs, mpi_elapsed);
		

	}




	return 0;
}