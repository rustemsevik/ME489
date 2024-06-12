#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda.h"

/* ************************************************************************* */
int Nd, Nc, Np;
double TOL;

#define BUFSIZE 512
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Define maximum number of iterations
#define MAX_ITER 10000

/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  // Error Check
  if (fp == NULL) {
    printf("Error opening the input file\n");
  }

  int sk = 0; double result;
  char buffer[BUFSIZE], fileTag[BUFSIZE];

  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result);
      sk++;
      return result;
    }
  }

  if(sk==0){
    printf("ERROR! Could not find the tag: [%s] in the file [%s]\n", tag, fileName);
    exit(EXIT_FAILURE);
  }
  return 0;
}

/* ************************************************************************** */
void readDataFile(char *fileName, double *data){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
  }

  int sk = 0;
  char buffer[BUFSIZE], fileTag[BUFSIZE];

  int shift = Nd;
  while(fgets(buffer, BUFSIZE, fp) != NULL){
      if(Nd==2)
        sscanf(buffer, "%lf %lf", &data[sk*shift + 0], &data[sk*shift+1]);
      if(Nd==3)
        sscanf(buffer, "%lf %lf %lf", &data[sk*shift + 0], &data[sk*shift+1], &data[sk*shift+2]);
      if(Nd==4)
        sscanf(buffer, "%lf %lf %lf %lf", &data[sk*shift+0],&data[sk*shift+1], &data[sk*shift+2], &data[sk*shift+3]);
      sk++;
  }
}

/* ************************************************************************** */
void writeDataToFile(char *fileName, double *data, int *Ci){
  FILE *fp = fopen(fileName, "w");
  if (fp == NULL) {
    printf("Error opening the output file\n");
  }

  for(int p=0; p<Np; p++){
    fprintf(fp, "%d %d ", p, Ci[p]);
    for(int dim=0; dim<Nd; dim++){
      fprintf(fp, "%.4f ", data[p*Nd + dim]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/* ************************************************************************** */
void writeCentroidToFile(char *fileName, double *Cm){
  FILE *fp = fopen(fileName, "w");
  if (fp == NULL) {
    printf("Error opening the output file\n");
  }

  for(int n=0; n<Nc; n++){
    for(int dim=0; dim<Nd; dim++){
      fprintf(fp, "%.4f ", Cm[n*Nd + dim]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/*************************************************************************** */
// Function to calculate Euclidean distance between two points
__device__ double distance(double *a, double *b, int Nd) {
  double sum = 0.0;
  for(int dim=0; dim < Nd; dim++){
    sum += pow((a[dim] - b[dim]), 2);
  }
  return sqrt(sum);
}

/* ************************************************************************* */
// Kernel to assign each point to the nearest centroid
__global__ void assignPointsKernel(double *data, int *Ci, int *Ck, double *Cm, int Np, int Nd, int Nc) {

    // map from thread and thread-block indices into linear array index
    int p = threadIdx.x + blockIdx.x * blockDim.x;


    if (p < Np) {
        double min_distance = INFINITY;
        int cluster_index   = 0;

        for (int n = 0; n < Nc; n++){
            double d = distance(&data[p * Nd], &Cm[n * Nd], Nd);
            if (d < min_distance) {
                min_distance  = d;
                cluster_index = n;
            }
        }

        atomicAdd(&Ck[cluster_index], 1);
        Ci[p] = cluster_index;
    }
}

/* ************************************************************************* */
// Kernel to update centroids based on the mean of assigned points
__global__ void sumPointsKernel(double *data, int *Ci, int *Ck, double *Cm, int Np, int Nd) {
    
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if(p < Np){
         // Get cluster of the point
        int cluster_index = Ci[p];
        
        for(int dim = 0; dim < Nd; dim++){
            atomicAdd(&Cm[cluster_index * Nd + dim], data[p * Nd + dim]);
        }
    }
}

/* ************************************************************************* */
// Function to update centroids based on the mean of assigned points
double updateCentroids(double *data, int *Ci, int *Ck, double *Cm, double *CmCopy, int Np, int Nd, int Nc) {

    double err = 1.E-12;
    for (int n = 0; n < Nc; n++){
        for(int dim = 0; dim<Nd; dim++){
        Cm[n*Nd + dim] = Cm[n*Nd + dim]/Ck[n];
        err = MAX(err, fabs(Cm[n*Nd + dim] - CmCopy[n*Nd + dim]));
        }
    }
    return err;
}

/* ************************************************************************* */
// Function to perform k-means clustering
void kMeans(double *data, int *Ci, int *Ck, double *Cm, double *CmCopy, int T) {
    // Initialize clusters randomly
    for (int n = 0; n < Nc; n++) {
        int ids = rand() % Np;
        for(int dim = 0; dim < Nd; dim++){
            Cm[n * Nd + dim] = data[ids * Nd + dim];
        }
        Ck[n] = 0;
        Ci[ids] = n;
    }

    double err = INFINITY;
    int sk = 0;

    while(err > TOL) {
        // Reset Ck array
        cudaMemset(Ck, 0, Nc * sizeof(int));
        assignPointsKernel<<<(Np + T - 1) / T, T>>>(data, Ci, Ck, Cm, Np, Nd, Nc);
        cudaDeviceSynchronize();

        // Copy Cm to CmCopy
        cudaMemcpy(CmCopy, Cm, Nc * Nd * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        // Reset Cm array
        cudaMemset(Cm, 0, Nc * Nd * sizeof(double));
        sumPointsKernel<<<(Np + T -1) / T, T>>>(data, Ci, Ck, Cm, Np, Nd);
        cudaDeviceSynchronize();

        err = updateCentroids(data, Ci, Ck, Cm, CmCopy, Np, Nd, Nc);
        // printf("\r Iteration %d %.12e\n", sk, err); sk++;
        fflush(stdout);
    }
    // printf("\n");
}

/* ************************************************************************* */

int main (int argc, char *argv[]) {
    if(argc != 3){
      printf("Usage: ./kmeans input.dat data.dat\n");
      return -1;
    }

    // Read Number of Data Points
    Np = (int) readInputFile(argv[1], "NUMBER_OF_POINTS");
    // Read Number of clusters
    Nc = (int) readInputFile(argv[1], "NUMBER_OF_CLUSTERS");
    // Read Dimension of Data
    Nd = (int) readInputFile(argv[1], "DATA_DIMENSION");
    // Read Tolerance
    TOL = readInputFile(argv[1], "TOLERANCE");

    // Allocate data [x_i, y_i, z_i, ...]
    double *data; cudaMallocManaged(&data, Np * Nd * sizeof(double));
    // Cluster id mapping every point to cluster
    int *Ci; cudaMallocManaged(&Ci, Np * sizeof(int));
    // Number of data points in every cluster
    int *Ck; cudaMallocManaged(&Ck, Nc * sizeof(int));
    // Centroid of every clusters
    double *Cm; cudaMallocManaged(&Cm, Nc * Nd * sizeof(double));
    // Copy of centroids to compute error
    double *CmCopy; cudaMallocManaged(&CmCopy, Nc * Nd * sizeof(double));

    // Fill point data from file
    readDataFile(argv[2], data);
    
    // Create events
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Number of threads per thread-block to use
    int T = 512;

    // Record start event
    cudaEventRecord(start);

    // Perform k-means clustering
    kMeans(data, Ci, Ck, Cm, CmCopy,T);

    // Insert end record event in stream
    cudaEventRecord(end);

    // Print out elapsed time
    float elapsed;
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    elapsed /= 1000.; // convert to seconds

    printf("Elapsed time: %g\n", elapsed);

    // Report Results
    for(int n = 0; n < Nc; n++){
      int Npoints = Ck[n];
      printf("(%d of %d) points are in the cluster %d with centroid( ", Npoints, Np, n);
       for(int dim = 0; dim < Nd; dim++){
               printf("%f ", Cm[n * Nd + dim]);
        }
      printf(") \n");

    }

    writeDataToFile("output.dat", data, Ci);
    writeCentroidToFile("centroids.dat", Cm);

    cudaFree(data);
    cudaFree(Ci);
    cudaFree(Ck);
    cudaFree(Cm);
    cudaFree(CmCopy);

    return 0;
}
