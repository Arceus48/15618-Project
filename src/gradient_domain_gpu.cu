#include <stdio.h>
#include <string.h>

#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "data_util.h"
#include "math_util_gpu.cu_inl"
using namespace std;

const int EDGE_WIDTH = 5;
#define THREADS_PER_BLK 1024
#define BLK_WIDTH 32

__global__ void array_subreduce(double *a, double *result, int nrow, int ncol) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrow || col >= ncol) {
        return;
    }
    int input_idx = row * ncol + col;
    int output_idx = blockIdx.y * gridDim.x + blockIdx.x;
    atomicAdd(result + output_idx, a[input_idx]);
}

void array_recursive_reduce(double *a, double *result, int nrow, int ncol) {
    while (nrow * ncol > THREADS_PER_BLK) {
        cudaMemset(result, 0, sizeof(double) * nrow * ncol);
        dim3 gridDim((ncol + BLK_WIDTH - 1) / BLK_WIDTH,
            (nrow + BLK_WIDTH - 1) / BLK_WIDTH);
        dim3 blockDim(BLK_WIDTH, BLK_WIDTH);

        array_subreduce<<gridDim, blockDim>>(a, result, nrow, ncol);

        nrow = (nrow + BLK_WIDTH - 1) / BLK_WIDTH;
        ncol = (ncol + BLK_WIDTH - 1) / BLK_WIDTH;
        cudaMemcpy(a, result, nrow * ncol * sizeof(double),
            cudaMemcpyDeviceToDevice);
    }

    cudaMemset(result, 0, sizeof(double) * nrow * ncol);
    dim3 gridDim(1, 1);
    dim3 blockDim(ncol, nrow);
    array_subreduce<<gridDim, blockDim>>(a, result, nrow, ncol);
}

double* tmp1;
double* tmp2;
double* tmp3;
double* nume;
double* denom;
double* M;
double* Ws;
double* one_M;
double* one_Ws;

double* B;
double* r;
double* tmp;
double* q;
double* d;
double* D;
double result;

/**
 * Allocate space for output before calling the function!
 *
 * @param[in] D shape: nrow * ncol
 * @param[in] input shape: nrow * ncol
 * @param[out] output shape: nrow * ncol
 * @param[in] nrow
 * @param[in] ncol
 * @param[in] conv
 * @param[in] niter
 */
void gradIntegrate(double* D, double* input, double* output, int nrow, int ncol,
                   double conv = 1e-3, int niter = 2000) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    B[i] = 0.0;
  }
#pragma omp for
  for (int i = EDGE_WIDTH; i < nrow - EDGE_WIDTH; i++) {
    for (int j = EDGE_WIDTH; j < ncol - EDGE_WIDTH; j++) {
      B[i * ncol + j] = 1.0;
    }
  }
#pragma omp for
  // Allocate space for output before calling the function!
  for (int i = 0; i < nrow * ncol; i++) {
    output[i] = input[i];
  }

  laplacian(output, r, nrow, ncol);

#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    r[i] = B[i] * (D[i] - r[i]);
  }

#pragma omp for
  // Allocate space for output before calling the function!
  for (int i = 0; i < nrow * ncol; i++) {
    d[i] = r[i];
  }

  result = 0;
#pragma omp barrier
#pragma omp for reduction(+ : result)
  for (int j = 0; j < nrow * ncol; j++) {
    result += r[j] * r[j];
  }
  double delta = result;
  double ita;
  double delta_old;
  double beta;
  for (int i = 0; i < niter; i++) {
    double loss = sqrt(delta);
    // #pragma omp single
    //     printf("Iter: %d, loss: %f\n", i, loss);
    if (loss <= conv) {
      break;
    }

    laplacian(d, q, nrow, ncol);

    result = 0;
#pragma omp barrier
#pragma omp for reduction(+ : result)
    for (int j = 0; j < nrow * ncol; j++) {
      result += d[j] * q[j];
    }
    ita = delta / result;

#pragma omp for nowait
    for (int j = 0; j < nrow * ncol; j++) {
      output[j] += B[j] * ita * d[j];
    }

#pragma omp for
    for (int j = 0; j < nrow * ncol; j++) {
      r[j] = B[j] * (r[j] - ita * q[j]);
    }

    delta_old = delta;

    result = 0;
#pragma omp barrier
#pragma omp for reduction(+ : result)
    for (int j = 0; j < nrow * ncol; j++) {
      result += r[j] * r[j];
    }
    delta = result;
    beta = delta / delta_old;

#pragma omp barrier
#pragma omp for
    for (int j = 0; j < nrow * ncol; j++) {
      d[j] = r[j] + beta * d[j];
    }
  }
}

/**
 * Need to allocate space for gradA, gradF, fused before calling this function!
 *
 * @param[in] A shape: nrow * ncol
 * @param[in] F shape: nrow * ncol
 * @param[in] sig
 * @param[in] thld
 * @param[out] gradA shape: 2 * nrow * ncol (first nrow * ncol is for x axis,
 * the remaining for y)
 * @param[out] gradF shape: 2 * nrow * ncol (first nrow * ncol is for x axis,
 * the remaining for y)
 * @param[out] fused shape: 2 * nrow * ncol (first nrow * ncol is for x axis,
 * the remaining for y)
 * @param[in] nrow
 * @param[in] ncol
 */
 __global__ void fuseGrad(double* A, double* F, double sig, double thld, double* gradA,
              double* gradF, double* fused, int nrow, int ncol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_idx = row * ncol + col;
    if (row <= 0 || row + 1 >= nrow || col <= 0 || col + 1 >= ncol) {
        return;
    }
  
    gradientx(A, gradA, ncol, linear_idx);
    gradienty(A, gradA + nrow * ncol, ncol, linear_idx);
    gradientx(F, gradF, ncol, linear_idx);
    gradienty(F, gradF + nrow * ncol, ncol, linear_idx);

    element_multiply(gradA, gradF, tmp1, nrow, ncol * 2);
    element_add(tmp1, tmp1 + nrow * ncol, tmp2, nrow, ncol);
    element_abs(tmp2, nume, nrow, ncol);

    element_multiply(gradA, gradA, tmp1, nrow, ncol * 2);
    element_add(tmp1, tmp1 + nrow * ncol, tmp2, nrow, ncol);
    element_multiply(gradF, gradF, tmp1, nrow, ncol * 2);
    element_add(tmp1, tmp1 + nrow * ncol, tmp3, nrow, ncol);
    element_multiply(tmp2, tmp3, denom, nrow, ncol);
    element_sqrt(denom, denom, nrow, ncol);

    // tmp2 holds magA ^ 2, tmp3 holds magF ^ 2
    element_divide_skip_0(nume, denom, M, nrow, ncol, 0);
    element_set_value_below_threshold(M, tmp2, nrow, ncol, 5e-3 * 5e-3, 1.0);
    element_set_value_below_threshold(M, tmp3, nrow, ncol, 5e-3 * 5e-3, 0.0);

    element_subtract(F, thld, tmp2, nrow, ncol);
    element_scale(tmp2, sig, Ws, nrow, ncol);
    element_tanh(Ws, nrow, ncol);

    element_add(Ws, 1.0, Ws, nrow, ncol);
    element_scale(Ws, 0.5, Ws, nrow, ncol);

    element_subtract(1.0, Ws, one_Ws, nrow, ncol);
    element_subtract(1.0, M, one_M, nrow, ncol);

    // Merge into fuse. Done half by half to avoid stacking.
    element_multiply(one_M, gradA, fused, nrow, ncol);
    element_multiply(M, gradF, tmp2, nrow, ncol);
    element_add(tmp2, fused, fused, nrow, ncol);
    element_multiply(fused, one_Ws, fused, nrow, ncol);
    element_multiply(Ws, gradA, tmp3, nrow, ncol);
    element_add(fused, tmp3, fused, nrow, ncol);

    element_multiply(one_M, gradA + nrow * ncol, fused + nrow * ncol, nrow, ncol);
    element_multiply(M, gradF + nrow * ncol, tmp2, nrow, ncol);
    element_add(tmp2, fused + nrow * ncol, fused + nrow * ncol, nrow, ncol);
    element_multiply(fused + nrow * ncol, one_Ws, fused + nrow * ncol, nrow,
                    ncol);
    element_multiply(Ws, gradA + nrow * ncol, tmp3, nrow, ncol);
    element_add(fused + nrow * ncol, tmp3, fused + nrow * ncol, nrow, ncol);
}

/**
 * Allocate space for gradA, gradF and fused before calling the function!
 * @param[in] A double*[3], each is a pointer to an array: nrow * ncol
 * @param[in] F double*[3], each is a pointer to an array: nrow * ncol
 * @param[out] gradA size: 3 * 2 * nrow * ncol
 * @param[out] gradF size: 3 * 2 * nrow * ncol
 * @param[out] fused size: 3 * 2 * nrow * ncol
 * @param[in] sig
 * @param[in] thld
 * @param[in] nrow
 * @param[in] ncol
 */
void fuseGradRgb(double** A, double** F, double* gradA, double* gradF,
                 double* fused, double sig, double thld, int nrow, int ncol) {
#pragma omp parallel default(shared)
  {
    for (int i = 2; i >= 0; i--) {
      fuseGrad(A[i], F[i], sig, thld, gradA + i * 2 * nrow * ncol,
               gradF + i * 2 * nrow * ncol, fused + i * 2 * nrow * ncol, nrow,
               ncol);
    }
  }
}

/**
 * Allocate space for output before calling the function!
 *
 * @param[in] fused size: 3 * 2 * nrow * ncol
 * @param[in] argb double*[3], each is a pointer to an array: nrow * ncol
 * @param[in] frgb double*[3], each is a pointer to an array: nrow * ncol
 * @param[out] output double*[3], each is a pointer to an array: nrow * ncol
 * @param[in] bound
 * @param[in] init_opt
 * @param[in] conv
 * @param[in] niter
 */
void gradInteRgb(double* fused, double** argb, double** frgb, double** output,
                 int bound, int init_opt, int nrow, int ncol,
                 double conv = 1e-3, int niter = 2000) {
  (void)argb;
  (void)bound;
  (void)init_opt;
  // Assume init_opt is 2 and bound is 2.
  // input pointer should be frgb.
#pragma omp parallel default(shared)
  {
    for (int i = 2; i >= 0; i--) {
      divergence(fused + i * 2 * nrow * ncol, fused + (i * 2 + 1) * nrow * ncol,
                 D, nrow, ncol);
      gradIntegrate(D, frgb[i], output[i], nrow, ncol, conv, niter);
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 9) {
    printf("Check params.\n");
    return 0;
  }
  // argv[1]: ambient photo path
  // argv[2]: flash photo path
  // use_raw and mode are ignored

  double sig = stod(argv[3]);
  double thld = stod(argv[4]);
  int bound_cond = stoi(argv[5]);
  int init_opt = stoi(argv[6]);
  double conv = stod(argv[7]);
  int niter = stoi(argv[8]);

  // Load the two images
  int nrow;
  int ncol;
  double** argb = readImage(argv[1], &nrow, &ncol);
  double** frgb = readImage(argv[2], &nrow, &ncol);

  // Allocate buffers
  double* gradA = new double[3 * 2 * nrow * ncol];
  double* gradF = new double[3 * 2 * nrow * ncol];
  double* fused = new double[3 * 2 * nrow * ncol];
  double** output = new double*[3];
  for (int i = 0; i < 3; i++) {
    output[i] = new double[nrow * ncol];
  }
  cout << nrow << " " << ncol << endl;
  // Buffers for other functions
  tmp1 = new double[nrow * ncol * 2];
  tmp2 = new double[nrow * ncol];
  tmp3 = new double[nrow * ncol];
  nume = new double[nrow * ncol];
  denom = new double[nrow * ncol];
  M = new double[nrow * ncol];
  Ws = new double[nrow * ncol];
  one_M = new double[nrow * ncol];
  one_Ws = new double[nrow * ncol];

  B = new double[nrow * ncol];
  r = new double[nrow * ncol];
  tmp = new double[nrow * ncol];
  q = new double[nrow * ncol];
  d = new double[nrow * ncol];
  D = new double[nrow * ncol];

  // Computation
  auto start = std::chrono::high_resolution_clock::now();
  fuseGradRgb(argb, frgb, gradA, gradF, fused, sig, thld, nrow, ncol);
  auto end1 = std::chrono::high_resolution_clock::now();
  gradInteRgb(fused, argb, frgb, output, bound_cond, init_opt, nrow, ncol, conv,
              niter);
  auto end = std::chrono::high_resolution_clock::now();

  printf("Part 1: %ld ms, Part 2: %ld ms\n",
         chrono::duration_cast<chrono::milliseconds>(end1 - start).count(),
         chrono::duration_cast<chrono::milliseconds>(end - end1).count());
  printf("Total time: %ld ms\n",
         chrono::duration_cast<chrono::milliseconds>(end - start).count());

  saveImage(output, "test.png", nrow, ncol);

  // Memory leak? Who cares!
  return 0;
}