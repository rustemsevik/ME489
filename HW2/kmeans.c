#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_ITER 10e+12

// Function to read the input parameters from the input file
void read_input_parameters(const char *filename, int *N, int *K, int *d, double *tolerance) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open input file");
        exit(EXIT_FAILURE);
    }
    
    fscanf(file, "[NUMBER_OF_POINTS]\n%d\n", N);
    fscanf(file, "[NUMBER_OF_CLUSTERS]\n%d\n", K);
    fscanf(file, "[DATA_DIMENSION]\n%d\n", d);
    fscanf(file, "[TOLERANCE]\n%lf\n", tolerance);
    
    fclose(file);
}

// Function to read data points from a .dat file
double** read_data_points(const char *filename, int N, int d) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open data file");
        exit(EXIT_FAILURE);
    }
    
    double **data_points = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        data_points[i] = (double *)malloc(d * sizeof(double));
        for (int j = 0; j < d; j++) {
            fscanf(file, "%lf", &data_points[i][j]);
        }
    }
    
    fclose(file);
    return data_points;
}

// Function to calculate the Euclidean distance between two points
double euclidean_distance(double *point1, double *point2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

void assign_clusters(double **data, double **centroids, int *clusters, int N, int K, int D, int *counts) {
    double min_dist, dist;
    int cluster;
    for (int i = 0; i < N; i++) {
        min_dist = INFINITY;
        cluster = -1; // Ensure cluster gets assigned
        for (int j = 0; j < K; j++) {
            dist = euclidean_distance(data[i], centroids[j], D);
            if (dist < min_dist) {
                min_dist = dist;
                cluster = j;
            }
        }
        clusters[i] = cluster;
        if (cluster >= 0 && cluster < K) {
            counts[cluster]++;  // Ensure this line is inside your cluster assignment function        
        }
    }
}

void update_centroids(double **data, int *clusters, double **centroids, int *counts, int N, int K, int d) {
    // Reset centroids and counts
    for (int i = 0; i < K; i++) {
        counts[i] = 0;
        for (int j = 0; j < d; j++) {
            centroids[i][j] = 0.0;
        }
    }

    // Sum up and count points for each cluster
    for (int i = 0; i < N; i++) {
        int cluster_idx = clusters[i];
        counts[cluster_idx]++;
        for (int j = 0; j < d; j++) {
            centroids[cluster_idx][j] += data[i][j];
        }
    }

    // Divide by count to get the new centroids
    for (int i = 0; i < K; i++) {
        if (counts[i] > 0) {
            for (int j = 0; j < d; j++) {
                centroids[i][j] /= counts[i];
            }
        }
    }
}

void median_based_init(double **data_points, double **centroids, int N, int K, int d) {   
    // Calculate the number of points in each segment
    int segment_size = N / K;
    
    for (int i = 0; i < K; i++) {
        // Calculate the index of the median point for the current segment
        int median_index = i * segment_size + segment_size / 2;
        
        // Handle the case where N is not perfectly divisible by K
        // by adjusting the last centroid position if necessary
        if (i == K - 1 && N % K != 0) {
            median_index = N - (segment_size / 2) - 1;
        }
        
        // Assign the median data point to the centroid
        for (int j = 0; j < d; j++) {
            centroids[i][j] = data_points[median_index][j];
        }
    }
}

// Function to check if centroids have converged
int has_converged(double **prev_centroids, double **centroids, int K, int d, double threshold) {
    for (int i = 0; i < K; i++) {
        if (euclidean_distance(prev_centroids[i], centroids[i], d) > threshold) {
            return 0; // Centroids have not converged
        }
    }
    return 1; // Centroids have converged
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <data_file>\n", argv[0]);
        return 1;
    }

    //srand(time(NULL)); // Seed the random number generator

    int N, K, d;
    double tolerance;
    // Read input parameters
    read_input_parameters(argv[1], &N, &K, &d, &tolerance);
    
    int *counts = calloc(K, sizeof(int));

    if (counts == NULL) {
        perror("Memory allocation for counts failed");
        exit(EXIT_FAILURE);
    }

    // Read data points
    double **data_points = read_data_points(argv[2], N, d);

    // Allocate memory for centroids and clusters
    double **centroids = malloc(K * sizeof(double *));

    double **prev_centroids = malloc(K * sizeof(double *)); 
    for (int i = 0; i < K; i++) {
        centroids[i] = malloc(d * sizeof(double));
        prev_centroids[i] = malloc(d * sizeof(double)); 
        
    }

    // Randomly initialize centroids by selecting random data points
    // for (int i = 0; i < K; i++) {
    //     int random_index = rand() % N; // Get a random index
    //     for (int j = 0; j < d; j++) {
    //         centroids[i][j] = data_points[random_index][j]; // Initialize centroid with a random data point}}

    //Initializing Method
    median_based_init(data_points, centroids, N, K, d);
    int *clusters = malloc(N * sizeof(int));

    // K-Means algorithm
    int iter = 0;
    int converged = 0;

    while (iter < MAX_ITER && !converged) {
        memset(counts, 0, K * sizeof(int));  // Reset counts at the beginning of each iteration
        assign_clusters(data_points, centroids, clusters, N, K, d, counts);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < d; j++) {
                prev_centroids[i][j] = centroids[i][j]; // Store current centroids as previous
            }
        }

        update_centroids(data_points, clusters, centroids, counts, N, K, d);

        converged = has_converged(prev_centroids, centroids, K, d, tolerance);

        iter++;
    }

    // Write results to output file
    FILE *output_file = fopen("output.dat", "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        fprintf(output_file, "%d %d", i, clusters[i]);
        for (int j = 0; j < d; j++) {
            fprintf(output_file, " %.4f", data_points[i][j]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);

    FILE *centroid_file = fopen("centroids.dat", "w");
    if (centroid_file == NULL) {
        perror("Error opening centroid output file");
        return 1;
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < d; j++) {
            fprintf(centroid_file, "%.4f ", centroids[i][j]);
        }
        fprintf(centroid_file, "\n");
    }
    fclose(centroid_file);

    // Print the number of points in every cluster and the centroid of every cluster
    for (int i = 0; i < K; i++) {
        printf("(%d of %d) points are in the cluster %d with centroid(", counts[i], N, i);
        for (int j = 0; j < d; j++) {
            printf("%.4f", centroids[i][j]);
            if (j < d - 1) printf(", ");
        }
        printf(")\n");
    }

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(data_points[i]);
    }
    free(data_points);
    for (int i = 0; i < K; i++) {
        free(centroids[i]);
    }
    free(centroids);
    free(clusters);
    free(counts);

    return 0;
}