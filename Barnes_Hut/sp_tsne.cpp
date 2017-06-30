/*
*

*/

#pragma warning(disable:4996)  

#define CUB_STDERR

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdint.h>

#include "sptree.h"
#include "sp_tsne.h"

using namespace std;

// Perform t-SNE
void TSNE::run(double* sorted_distances, int* sorted_indices, int N, int no_dims, int K,
	int perplexity, double theta, double eta, int iterations, int verbose, double* Y) {

	setbuf(stdout, NULL);
	//setvbuf(stdout, NULL, _IONBF, 1024);

	// Determine whether we are using an exact algorithm
	if (N - 1 < 3 * perplexity) {
		printf("Perplexity ( = %i) too large for the number of data points (%i)!\n", perplexity, N);
		exit(1);
	}
	if (verbose > 0) printf("Using no_dims = %d, perplexity = %d, learning rate = %f, and theta = %f\n", no_dims, perplexity, eta, theta);



	// Set learning parameters
	float total_time = .0;
	clock_t start, end;
	int max_iter = iterations, stop_lying_iter = 250, mom_switch_iter = 250;
	double momentum = .5;
	double final_momentum = .8;
	float exageration = 12.0;

	int* col_P = NULL;
	double* val_P = NULL;
	int* row_P = NULL;

	row_P = (int*)malloc((N + 1) * sizeof(int));
	if (*row_P == NULL) { printf("Memory allocation failed 2!\n"); exit(1); }
	row_P[0] = 0;
	for (int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (int)K;

	// Allocate some memory
	double* dY = (double*)malloc(N * no_dims * sizeof(double));
	double* uY = (double*)malloc(N * no_dims * sizeof(double));
	double* gains = (double*)malloc(N * no_dims * sizeof(double));
	if (dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed 1!\n"); exit(1); }
	for (int i = 0; i < N * no_dims; i++)    uY[i] = .0;
	for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;
;

	double* cur_P = (double*)malloc((N - 1) * sizeof(double));
	if (cur_P == NULL) { printf("Memory allocation failed 3!\n"); exit(1); }


	computeGaussianPerplexity(sorted_distances, sorted_indices, N, K, perplexity, &row_P, &col_P, &val_P);  // computing all distances

	// Symmetrize input similarities
	symmetrizeMatrix(&row_P, &col_P, &val_P, N);
	double sum_P = .0;
	for (int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
	for (int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;


	// Lie about the P-values
	for (int i = 0; i < row_P[N]; i++)		val_P[i] *= exageration;

	// Initialize solution (randomly)
	for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;


	// Perform main training loop
	if (verbose > 0) printf("\nLearning embedding...\n");
	start = clock();
	for (int iter = 0; iter < max_iter; iter++) {

		// Compute (approximate) gradient
		computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta);

		// Update gains
		//for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
		for (int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .05) : (gains[i] * .95);
		for (int i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

		// Perform gradient update (with momentum and gains)
		for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for (int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

		// Make solution zero-mean
		zeroMean(Y, N, no_dims);

		// Stop lying about the P-values after a while, and switch momentum
		if (iter == stop_lying_iter) {
			for (unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= exageration;
		}
		if (iter == mom_switch_iter) momentum = final_momentum;

		// Save tSNE progress after each iteration
		if (verbose > 2)
		{
			// Open file, write first 2 integers and then the data
			FILE *h;
			char interim_filename[_MAX_PATH];
			sprintf_s(interim_filename, "interim_%06i.dat", iter);
			fopen_s(&h, interim_filename, "w + b");
			if (h == NULL)
			{
				printf("Error: could not open data file.\n");
				return;
			}
			fwrite(&N, sizeof(int), 1, h);
			fwrite(&no_dims, sizeof(int), 1, h);
			fwrite(Y, sizeof(double), N * no_dims, h);
			fclose(h);
		}

		// Print out progress
		if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
			end = clock();
			double C = .0;
			C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
			if (iter == 0) {
				if (verbose > 1) printf("Iteration %d: error is %f\n", iter + 1, C);
			}
			else {
				total_time += (float)(end - start) / CLOCKS_PER_SEC;
				if (verbose > 1) printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start) / CLOCKS_PER_SEC);
			}
			start = clock();
		}
	}
	end = clock(); total_time += (float)(end - start) / CLOCKS_PER_SEC;

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);
	free(row_P); row_P = NULL;
	free(col_P); col_P = NULL;
	free(val_P); val_P = NULL;

	if (verbose > 0) printf("Fitting performed in %4.2f seconds.\n", total_time);
}


void TSNE::computeGaussianPerplexity(double* sorted_distances, int* sorted_indices, int N, int K, double perplexity, int** _row_P, int** _col_P, double** _val_P) {
	float start;
	float end;

	// Allocate the memory we need
	*_row_P = (int*)malloc((N + 1) * sizeof(int));
	*_col_P = (int*)calloc(N * K, sizeof(int));
	*_val_P = (double*)calloc(N * K, sizeof(double));
	if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	int* row_P = *_row_P;
	int* col_P = *_col_P;
	double* val_P = *_val_P;
	double* cur_P = (double*)malloc((N - 1) * sizeof(double));
	if (cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	row_P[0] = 0;
	for (int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (int)(K - 1);

	double cur_dist;
	int cur_index;


	for (int n = 0; n < N; n++) {

		// Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while (!found && iter < 200) {

			// --- HERE is the problem!
			// Compute Gaussian kernel row
			for (int m = 0; m < K; m++)
			{
				cur_dist = sorted_distances[n*K + m + 1];
				cur_P[m] = exp(-beta * cur_dist);
			}

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < K; m++) sum_P += cur_P[m];
			double H = .0;
			for (int m = 0; m < K; m++)
			{
				cur_dist = sorted_distances[n*K + m + 1];
				H += beta * (cur_dist * cur_P[m]);
			}
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize current row of P and store in matrix
		for (int m = 0; m < K; m++) cur_P[m] /= sum_P;
		for (int m = 0; m < K; m++)
		{
			cur_index = sorted_indices[n*K + m + 1];
			col_P[row_P[n] + m] = cur_index;
			val_P[row_P[n] + m] = cur_P[m];
		}
	}

	// Clean up memory
	free(cur_P);
}



// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(int* _row_P, int* _col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{
	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;
	double* pos_f = (double*)calloc(N * D, sizeof(double));
	double* neg_f = (double*)calloc(N * D, sizeof(double));
	if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed 4!\n"); exit(1); }

	unsigned int* inp_row_P = reinterpret_cast<unsigned int*>(_row_P);
	unsigned int* inp_col_P = reinterpret_cast<unsigned int*>(_col_P);
	tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
	for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	for (int i = 0; i < N * D; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double* buff = (double*)calloc(D, sizeof(double));
	double sum_Q = .0;
	for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
	}

	// Clean up memory
	free(buff);
	delete tree;
	return C;
}

// Symmetrizes a sparse matrix
void TSNE::symmetrizeMatrix(int** _row_P, int** _col_P, double** _val_P, int N) {

	// Get sparse matrix
	int* row_P = *_row_P;
	int* col_P = *_col_P;
	double* val_P = *_val_P;

	// Count number of elements and row counts of symmetric matrix
	int* row_counts = (int*)calloc(N, sizeof(int));
	if (row_counts == NULL) { printf("Memory allocation failed 5!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) present = true;
			}
			if (present) row_counts[n]++;
			else {
				row_counts[n]++;
				row_counts[col_P[i]]++;
			}
		}
	}
	int no_elem = 0;
	for (int n = 0; n < N; n++) no_elem += row_counts[n];

	// Allocate memory for symmetrized matrix
	int* sym_row_P = (int*)malloc((N + 1) * sizeof(int));
	int* sym_col_P = (int*)malloc(no_elem * sizeof(int));
	double* sym_val_P = (double*)malloc(no_elem * sizeof(double));
	if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed 6!\n"); exit(1); }

	// Construct new row indices for symmetric matrix
	sym_row_P[0] = 0;
	for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (int)row_counts[n];

	// Fill the result matrix
	int* offset = (int*)calloc(N, sizeof(int));
	if (offset == NULL) { printf("Memory allocation failed 7!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

																								  // Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) {
					present = true;
					if (n <= col_P[i]) {                                                 // make sure we do not add elements twice
						sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
						sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
						sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
						sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
					}
				}
			}


			// If (col_P[i], n) is not present, there is no addition involved
			if (!present) {
				sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
				sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
				sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
				sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
			}


			// Update offsets
			if (!present || (present && n <= col_P[i])) {
				offset[n]++;
				if (col_P[i] != n) offset[col_P[i]]++;
			}
		}
	}

	// Divide the result by two
	for (int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

	// Return symmetrized matrices
	free(*_row_P); *_row_P = sym_row_P;
	free(*_col_P); *_col_P = sym_col_P;
	free(*_val_P); *_val_P = sym_val_P;

	// Free up some memery
	free(offset); offset = NULL;
	free(row_counts); row_counts = NULL;
}


// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*)calloc(D, sizeof(double));
	if (mean == NULL) { printf("Memory allocation failed 8!\n"); exit(1); }
	int nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
		nD += D;
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double)N;
	}

	// Subtract data mean
	nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
		nD += D;
	}
	free(mean); mean = NULL;
}

void TSNE::normalize(double* X, int N, int D) {
	double max_X = .0;
	for (int i = 0; i < N * D; i++) {
		if (X[i] > max_X) max_X = X[i];
	}
	for (int i = 0; i < N * D; i++) X[i] /= max_X;
}


// Generates a Gaussian random number
double TSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** sorted_distances, int** sorted_indices, int* n, int* no_dims, int* k, 
	int* perplexity, double* theta, double* eta, int* iterations, int* verbose) {

	// Open file, read first 2 integers, allocate memory, and read the data
	FILE *h;
	h = fopen("data.dat", "r+b");
	if (h == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}


	fread(theta, sizeof(double), 1, h);										// gradient accuracy
	fread(eta, sizeof(double), 1, h);										// eta (learning rate)

	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(no_dims, sizeof(int), 1, h);										// output dimensionality

	fread(k, sizeof(int), 1, h);											// Knns dimensionality
	
	fread(iterations, sizeof(int), 1, h);									// number of iterations
	fread(verbose, sizeof(int), 1, h);										// verbosity (between 0 and 3)

	fread(perplexity, sizeof(int), 1, h);								// perplexity
	

	*sorted_distances = (double*)malloc(*k * *n * sizeof(double));
	if (*sorted_distances == NULL) { printf("Memory allocation failed data!\n"); exit(1); }
	int sizeread = fread(*sorted_distances, sizeof(double), *n * *k, h);           // the Knns

	*sorted_indices = (int*)malloc(*k * *n * sizeof(int));
	if (*sorted_indices == NULL) { printf("Memory allocation failed col_p!\n"); exit(1); }
	sizeread = fread(*sorted_indices, sizeof(int), *n * *k, h);                               // the indices of the Knns


	fclose(h);
	if (*verbose > 0) printf("Read the %i x %i data matrix successfully!\n", *n, *k);
	return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(double* data, double* costs, int n, int d, int verbose) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	fopen_s(&h, "result.dat", "w+b");
	if (h == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n * d, h);
	fwrite(costs, sizeof(double), n, h);
	fclose(h);
	if (verbose > 0) printf("Wrote the %i x %i data matrix successfully!\n\n", n, d);
}


void TSNE::save_data(double* data, int n, int d, int verbose) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	fopen_s(&h, "result.dat", "w+b");
	if (h == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n * d, h);
	fclose(h);
	if (verbose > 0) printf("Wrote the %i x %i data matrix successfully!\n\n", n, d);
}

int main()
{
	int N, D, no_dims, k, iterations, perplexity;
	double theta, eta;
	double *sorted_distances = NULL; ;
	int *sorted_indices = NULL;
	int verbose;

	TSNE* tsne = new TSNE();

	time_t start = clock();
	// Read the parameters and the dataset
	if (tsne->load_data(&sorted_distances, &sorted_indices, &N, &no_dims, &k, &perplexity, &theta, &eta, &iterations, &verbose)) {
		
		double* Y = (double*)malloc(N * no_dims * sizeof(double));
		if (Y == NULL) { printf("Memory allocation failed Y!\n"); exit(1); }

		// Now fire up the SNE implementation
		//double* costs = (double*)calloc(N, sizeof(double));
		//if (costs == NULL) { printf("Memory allocation failed costs\n"); exit(1); }
		tsne->run(sorted_distances, sorted_indices, N, no_dims, k, perplexity, theta, eta, iterations, verbose, Y);
	
		// Save the results
		tsne->save_data(Y, N, no_dims, verbose);

		// Clean up the memory
		free(Y); Y = NULL;
	}
	delete(tsne);
	time_t end = clock();
	if (verbose > 0) printf("T-sne required %f seconds (%f minutes) to run\n", float(end - start) / CLOCKS_PER_SEC, float(end - start) / (60 * CLOCKS_PER_SEC));
}
