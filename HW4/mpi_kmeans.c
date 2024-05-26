#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define BUFSIZE 512
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX_ITER 10000

int Nd, Nc, Np;
double TOL;

double readInputFile(char *fileName, char* tag) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int sk = 0;
    double result;
    char buffer[BUFSIZE], fileTag[BUFSIZE];

    while (fgets(buffer, BUFSIZE, fp) != NULL) {
        sscanf(buffer, "%s", fileTag);
        if (strstr(fileTag, tag)) {
            fgets(buffer, BUFSIZE, fp);
            sscanf(buffer, "%lf", &result);
            sk++;
            return result;
        }
    }

    if (sk == 0) {
        printf("ERROR! Could not find the tag: [%s] in the file [%s]\n", tag, fileName);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 0;
}

void readDataFile(char *fileName, double *data) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int sk = 0;
    char buffer[BUFSIZE];

    int shift = Nd;
    while (fgets(buffer, BUFSIZE, fp) != NULL) {
        if (Nd == 2)
            sscanf(buffer, "%lf %lf", &data[sk * shift + 0], &data[sk * shift + 1]);
        if (Nd == 3)
            sscanf(buffer, "%lf %lf %lf", &data[sk * shift + 0], &data[sk * shift + 1], &data[sk * shift + 2]);
        if (Nd == 4)
            sscanf(buffer, "%lf %lf %lf %lf", &data[sk * shift + 0], &data[sk * shift + 1], &data[sk * shift + 2], &data[sk * shift + 3]);
        sk++;
    }
}

void writeDataToFile(char *fileName, double *data, int *Ci) {
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening the output file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int p = 0; p < Np; p++) {
        fprintf(fp, "%d %d ", p, Ci[p]);
        for (int dim = 0; dim < Nd; dim++) {
            fprintf(fp, "%.4f ", data[p * Nd + dim]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void writeCentroidToFile(char *fileName, double *Cm) {
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening the output file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int n = 0; n < Nc; n++) {
        for (int dim = 0; dim < Nd; dim++) {
            fprintf(fp, "%.4f ", Cm[n * Nd + dim]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

double distance(double *a, double *b) {
    double sum = 0.0;
    for (int dim = 0; dim < Nd; dim++) {
        sum += pow((a[dim] - b[dim]), 2);
    }
    return sqrt(sum);
}

void assignPoints(double *data, int *Ci, int *Ck, double *Cm, int start, int end) {
    for (int n = 0; n < Nc; n++) {
        Ck[n] = 0;
    }

    for (int p = start; p < end; p++) {
        double min_distance = INFINITY;
        int cluster_index = 0;

        for (int n = 0; n < Nc; n++) {
            double d = distance(&data[p * Nd], &Cm[n * Nd]);
            if (d < min_distance) {
                min_distance = d;
                cluster_index = n;
            }
        }

        Ck[cluster_index]++;
        Ci[p] = cluster_index;
    }
}

double updateCentroids(double *data, int *Ci, int *Ck, double *Cm, int start, int end) {
    double *CmCopy = (double *)malloc(Nc * Nd * sizeof(double));
    double *localCm = (double *)calloc(Nc * Nd, sizeof(double));
    int *localCk = (int *)calloc(Nc, sizeof(int));

    for (int n = 0; n < Nc; n++) {
        // printf("Centroid %d: \n", n);
        for (int dim = 0; dim < Nd; dim++) {
            // printf("%f ", Cm[n * Nd + dim]);
            CmCopy[n * Nd + dim] = Cm[n * Nd + dim];
            Cm[n * Nd + dim] = 0.0;
        }
    }

    for (int p = start; p < end; p++) {
        int cluster_index = Ci[p];

        for (int dim = 0; dim < Nd; dim++) {
            localCm[cluster_index * Nd + dim] += data[p * Nd + dim];
        }
        localCk[cluster_index]++;
    }

    // We use MPI_Allreduce to aggregate the local sums (localCm) and counts
    // (localCk) from all processors into global sums (Cm) and counts (Ck).
    MPI_Allreduce(localCm, Cm, Nc * Nd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(localCk, Ck, Nc, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    double err = 1.E-12;
    for (int n = 0; n < Nc; n++) {
        for (int dim = 0; dim < Nd; dim++) {
            Cm[n * Nd + dim] = Cm[n * Nd + dim] / Ck[n];
            err = MAX(err, fabs(Cm[n * Nd + dim] - CmCopy[n * Nd + dim]));
        }
    }

    free(CmCopy);
    free(localCm);
    free(localCk);
    return err;
}

void kMeans(double *data, int *Ci, int *Ck, double *Cm, int start, int end, int rank, int size) {
    if (rank == 0) {
        for (int n = 0; n < Nc; n++) {
            int ids = rand() % Np;  
            for (int dim = 0; dim < Nd; dim++) {
                Cm[n * Nd + dim] = data[ids * Nd + dim];
            }
            Ck[n] = 0;
        }
    }

    // Broadcast initial centroids to all processes
    MPI_Bcast(Cm, Nc * Nd, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    double err = INFINITY;
    int sk = 0;

    while (err > TOL) {
        assignPoints(data, Ci, Ck, Cm, start, end);
        double local_err = updateCentroids(data, Ci, Ck, Cm, start, end);
        MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            // printf("\n \r Iteration %d %.12e\n", sk, err);
            fflush(stdout);
        }
        sk++;
    }
    if (rank == 0) printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: mpiexec -np #number-of-processors# ./mpi_kmeans input.dat data.dat\n");
        }
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        Np = (int)readInputFile(argv[1], "NUMBER_OF_POINTS");
        Nc = (int)readInputFile(argv[1], "NUMBER_OF_CLUSTERS");
        Nd = (int)readInputFile(argv[1], "DATA_DIMENSION");
        TOL = readInputFile(argv[1], "TOLERANCE");
    }

    // Broadcast the input parameters to all processes
    MPI_Bcast(&Np, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nd, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&TOL, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *data = (double *)malloc(Np * Nd * sizeof(double));
    int *Ci = (int *)calloc(Np, sizeof(int));
    int *Ck = (int *)calloc(Nc, sizeof(int));
    double *Cm = (double *)calloc(Nc * Nd, sizeof(double));
    int *global_Ci = (int *)calloc(Np, sizeof(int));


    if (rank == 0) {
        readDataFile(argv[2], data);
    }

    MPI_Bcast(data, Np * Nd, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int start = (rank * Np) / size;
    int end = ((rank + 1) * Np) / size;

    double start_time = MPI_Wtime();
    

    kMeans(data, Ci, Ck, Cm, start, end, rank, size);
    // Calculate counts and displacements for each process
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            recvcounts[i] = ((i + 1) * Np / size) - (i * Np / size);
            displs[i] = i * Np / size;
        }
    }

    MPI_Gatherv(Ci + start, end - start, MPI_INT, global_Ci, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    double compute_time = end_time - start_time;

    if (rank == 0) {
        printf("Total Compute Time: %.12e using %d processors\n", compute_time, size);
        writeDataToFile("output.dat", data, global_Ci);
        writeCentroidToFile("centroids.dat", Cm);
        // Report Results
        for (int n = 0; n < Nc; n++) {
            int Npoints = Ck[n];
            printf("(%d of %d) points are in the cluster %d with centroid( ", Npoints, Np, n);
            for (int dim = 0; dim < Nd; dim++) {
                printf("%f ", Cm[n * Nd + dim]);
            }
            printf(")\n");
    }
    }

    free(data);
    free(Ci);
    free(Ck);
    free(Cm);

    MPI_Finalize();
    return 0;
}