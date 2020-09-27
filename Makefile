INC="inc"
NVCCFLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall
LFLAGS=-lglut -lGL
MP = mpicc

all: kmeans kmeans_cuda kmeans_mpi

kmeans: kmeans.c
	$(CC) kmeans.c -o kmeans
	
kmeans_cuda: kmeans_cuda.cu
	$(NVCC) $(NVCCFLAGS) kmeans_cuda.cu -o kmeans_cuda $(LFLAGS)

kmeans_mpi: kmeans_mpi.c
	$(MP) kmeans_mpi.c -o kmeans_mpi

clean:
	rm -vf kmeans kmeans_cuda kmeans_mpi
