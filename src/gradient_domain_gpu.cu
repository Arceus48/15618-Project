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
    // int tid = threadIdx.y * blockDim.x + threadIdx.x;
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

        array_subreduce<<<gridDim, blockDim>>>(a, result, nrow, ncol);

        nrow = (nrow + BLK_WIDTH - 1) / BLK_WIDTH;
        ncol = (ncol + BLK_WIDTH - 1) / BLK_WIDTH;
        cudaMemcpy(a, result, nrow * ncol * sizeof(double),
            cudaMemcpyDeviceToDevice);
    }

    cudaMemset(result, 0, sizeof(double) * nrow * ncol);
    dim3 gridDim(1, 1);
    dim3 blockDim(ncol, nrow);
    array_subreduce<<<gridDim, blockDim>>>(a, result, nrow, ncol);
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

// Grad Integ init stage kernel.
__global__ void init_kernel(double *gradX, double *gradY, double* D, double* input,
    double* output, double *B, double *r,
    double *d, double *tmp, int nrow, int ncol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_idx = row * ncol + col;
    if (row >= nrow || col >= ncol) {
        return;
    }

    bool is_boundary = (row == 0 || row + 1 == nrow
        || col == 0 || col + 1 == ncol);
    if (is_boundary) {
        D[linear_idx] = 0.0;
    } else {
        divergence(gradX, gradY, D, ncol, linear_idx);
    }

    B[linear_idx] = 0.0;
    if (row >= EDGE_WIDTH && row < nrow - EDGE_WIDTH
        && col >= EDGE_WIDTH && col < ncol - EDGE_WIDTH) {
        B[linear_idx] = 1.0;
    }

    // Allocate space for output before calling the function!
    output[linear_idx] = input[linear_idx];
    // memcpy(output, input, sizeof(double) * nrow * ncol);

    // No laplacian for boundary pixels.
    if (is_boundary) {
        r[linear_idx] = 0.0;
        d[linear_idx] = 0.0;
        tmp[linear_idx] = 0.0;
        return;
    }
    
    laplacian(input, r, ncol, linear_idx);
    element_subtract(D, r, r, linear_idx);
    element_multiply(B, r, r, linear_idx);
    
    d[linear_idx] = r[linear_idx];
    // memcpy(d, r, sizeof(double) * nrow * ncol);
    
    element_multiply(r, r, tmp, linear_idx);
}
// Grad Integ stage 1 kernel.
__global__ void inte_stage1(
    double *d, double *q, double *tmp, int nrow, int ncol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_idx = row * ncol + col;
    if (row >= nrow || col >= ncol) {
        return;
    }

    bool is_boundary = (row == 0 || row + 1 == nrow
        || col == 0 || col + 1 == ncol);

    if (is_boundary) {
        q[linear_idx] = 0.0;
        tmp[linear_idx] = 0.0;
        return;
    }

    laplacian(d, q, ncol, linear_idx);
    element_multiply(d, q, tmp, linear_idx);
}

// Grad Integ stage 2 kernel.
__global__ void inte_stage2(double* B, double *d, double *tmp,
    double *output, double *q, double *r,
    double ita, int nrow, int ncol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_idx = row * ncol + col;
    if (row >= nrow || col >= ncol) {
        return;
    }
    element_multiply(B, d, tmp, linear_idx);
    element_scale(tmp, ita, tmp, linear_idx);
    element_add(output, tmp, output, linear_idx);

    element_scale(q, ita, tmp, linear_idx);
    element_subtract(r, tmp, tmp, linear_idx);
    element_multiply(B, tmp, r, linear_idx);

    element_multiply(r, r, tmp, linear_idx);
}

// Grad Integ stage 3 kernel.
__global__ void inte_stage3(double *d, double *tmp,
    double *r, double beta, int nrow, int ncol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_idx = row * ncol + col;
    if (row >= nrow || col >= ncol) {
        return;
    }

    element_scale(d, beta, tmp, linear_idx);
    element_add(r, tmp, d, linear_idx);
}

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
void gradIntegrate(double *gradX, double *gradY,
    double* D, double* input, double* output, int nrow, int ncol,
    double conv = 1e-3, int niter = 2000) {
    dim3 gridDim((ncol + BLK_WIDTH - 1) / BLK_WIDTH,
    (nrow + BLK_WIDTH - 1) / BLK_WIDTH);
    dim3 blockDim(BLK_WIDTH, BLK_WIDTH);

    init_kernel<<<gridDim, blockDim>>>(gradX, gradY, D, input, output,
        B, r, d, tmp, nrow, ncol);

    double delta;
    double ita;
    double delta_old;
    double beta;

    array_recursive_reduce(tmp, tmp1, nrow, ncol);
    cudaMemcpy(&delta, tmp1, sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < niter; i++) {
        double loss = sqrt(delta);
    // #pragma omp single
        printf("Iter: %d, loss: %f\n", i, loss);
        if (loss <= conv) {
            break;
        }

        inte_stage1<<<gridDim, blockDim>>>(d, q, tmp, nrow, ncol);

        array_recursive_reduce(tmp, tmp1, nrow, ncol);
        cudaMemcpy(&ita, tmp1, sizeof(double), cudaMemcpyDeviceToHost);
        ita = delta / ita;

        inte_stage2<<<gridDim, blockDim>>>(B, d, tmp,
            output, q, r, ita, nrow, ncol);
        array_recursive_reduce(tmp, tmp1, nrow, ncol);
        delta_old = delta;
        cudaMemcpy(&delta, tmp1, sizeof(double), cudaMemcpyDeviceToHost);
        beta = delta / delta_old;
        inte_stage3<<<gridDim, blockDim>>>(d, tmp, r, beta, nrow, ncol);
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
              double* gradF, double* fused,
              double* tmp1, double* tmp2, double* tmp3,
              double* nume, double *denom, double* M, double* Ws,
              double* one_M, double* one_Ws,
              int nrow, int ncol) {
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

    element_multiply(gradA, gradF, tmp1, linear_idx);
    element_multiply(gradA + nrow * ncol, gradF + nrow * ncol,
        tmp1 + nrow * ncol, linear_idx);
    element_add(tmp1, tmp1 + nrow * ncol, tmp2, linear_idx);
    element_abs(tmp2, nume, linear_idx);

    element_multiply(gradA, gradA, tmp2, linear_idx);
    element_multiply(gradA + nrow * ncol, gradA + nrow * ncol,
        tmp2 + nrow * ncol, linear_idx);
    element_add(tmp2, tmp2 + nrow * ncol, tmp2, linear_idx);
    element_multiply(gradF, gradF, tmp3, linear_idx);
    element_multiply(gradF + nrow * ncol, gradF + nrow * ncol,
        tmp3 + nrow * ncol, linear_idx);
    element_add(tmp3, tmp3 + nrow * ncol, tmp3, linear_idx);
    element_multiply(tmp2, tmp3, denom, linear_idx);
    element_sqrt(denom, denom, linear_idx);

    // tmp2 holds magA ^ 2, tmp3 holds magF ^ 2
    element_divide_skip_0(nume, denom, M, linear_idx, 0);
    element_set_value_below_threshold(M, tmp2, linear_idx, 5e-3 * 5e-3, 1.0);
    element_set_value_below_threshold(M, tmp3, linear_idx, 5e-3 * 5e-3, 0.0);

    element_subtract(F, thld, tmp2, linear_idx);
    element_scale(tmp2, sig, Ws, linear_idx);
    element_tanh(Ws, linear_idx);

    element_add(Ws, 1.0, Ws, linear_idx);
    element_scale(Ws, 0.5, Ws, linear_idx);

    element_subtract(1.0, Ws, one_Ws, linear_idx);
    element_subtract(1.0, M, one_M, linear_idx);

    // Merge into fuse. Done half by half to avoid stacking.
    element_multiply(one_M, gradA, fused, linear_idx);
    element_multiply(M, gradF, tmp2, linear_idx);
    element_add(tmp2, fused, fused, linear_idx);
    element_multiply(fused, one_Ws, fused, linear_idx);
    element_multiply(Ws, gradA, tmp3, linear_idx);
    element_add(fused, tmp3, fused, linear_idx);

    element_multiply(one_M, gradA + nrow * ncol, fused + nrow * ncol, linear_idx);
    element_multiply(M, gradF + nrow * ncol, tmp2, linear_idx);
    element_add(tmp2, fused + nrow * ncol, fused + nrow * ncol, linear_idx);
    element_multiply(fused + nrow * ncol, one_Ws, fused + nrow * ncol, linear_idx);
    element_multiply(Ws, gradA + nrow * ncol, tmp3, linear_idx);
    element_add(fused + nrow * ncol, tmp3, fused + nrow * ncol, linear_idx);
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
    dim3 gridDim((ncol + BLK_WIDTH - 1) / BLK_WIDTH,
        (nrow + BLK_WIDTH - 1) / BLK_WIDTH);
    dim3 blockDim(BLK_WIDTH, BLK_WIDTH);
    for (int i = 2; i >= 0; i--) {
      fuseGrad<<<gridDim, blockDim>>>(A[i], F[i], sig, thld, gradA + i * 2 * nrow * ncol,
               gradF + i * 2 * nrow * ncol, fused + i * 2 * nrow * ncol,
               tmp1, tmp2, tmp3, nume, denom, M, Ws, one_M, one_Ws,
               nrow, ncol);
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
    for (int i = 2; i >= 0; i--) {
      gradIntegrate(fused + i * 2 * nrow * ncol,
        fused + (i * 2 + 1) * nrow * ncol,
        D, frgb[i], output[i],
        nrow, ncol, conv, niter);
    }
}

// void printGrad(double *fused, int nrow, int ncol) {
//     double **fuse_gradX = 
// }

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
  double** argb_gpu = new double*[3];
  double** frgb_gpu = new double*[3];
  for (int i = 0; i < 3; ++i) {
    cudaMalloc(argb_gpu + i, nrow * ncol * sizeof(double));
    cudaMalloc(frgb_gpu + i, nrow * ncol * sizeof(double));
    cudaMemcpy(argb_gpu[i], argb[i],
        nrow * ncol * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(frgb_gpu[i], frgb[i],
        nrow * ncol * sizeof(double), cudaMemcpyHostToDevice);
  }

  // Allocate buffers
  double *gradA, *gradF, *fused;
  cudaMalloc(&gradA, 3 * 2 * nrow * ncol * sizeof(double));
  cudaMalloc(&gradF, 3 * 2 * nrow * ncol * sizeof(double));
  cudaMalloc(&fused, 3 * 2 * nrow * ncol * sizeof(double));
  double** output = new double*[3];
  for (int i = 0; i < 3; i++) {
    output[i] = new double[nrow * ncol];
  }
  double** output_gpu = new double*[3];
  for (int i = 0; i < 3; i++) {
    cudaMalloc(output_gpu + i, nrow * ncol * sizeof(double));
  }
  cout << nrow << " " << ncol << endl;
  // Buffers for other functions
  cudaMalloc(&tmp1, nrow * ncol * 2 * sizeof(double));
  cudaMalloc(&tmp2, nrow * ncol * sizeof(double));
  cudaMalloc(&tmp3, nrow * ncol * sizeof(double));
  cudaMalloc(&nume, nrow * ncol * sizeof(double));
  cudaMalloc(&denom, nrow * ncol * sizeof(double));
  cudaMalloc(&M, nrow * ncol * sizeof(double));
  cudaMalloc(&Ws, nrow * ncol * sizeof(double));
  cudaMalloc(&one_M, nrow * ncol * sizeof(double));
  cudaMalloc(&one_Ws, nrow * ncol * sizeof(double));
  cudaMalloc(&B, nrow * ncol * sizeof(double));
  cudaMalloc(&r, nrow * ncol * sizeof(double));
  cudaMalloc(&tmp, nrow * ncol * sizeof(double));
  cudaMalloc(&q, nrow * ncol * sizeof(double));
  cudaMalloc(&d, nrow * ncol * sizeof(double));
  cudaMalloc(&D, nrow * ncol * sizeof(double));

  // Computation
  auto start = std::chrono::high_resolution_clock::now();
  fuseGradRgb(argb_gpu, frgb_gpu, gradA, gradF, fused, sig, thld, nrow, ncol);
  auto end1 = std::chrono::high_resolution_clock::now();
  gradInteRgb(fused, argb, frgb, output_gpu, bound_cond, init_opt, nrow, ncol, conv,
              niter);
  auto end = std::chrono::high_resolution_clock::now();

  printf("Part 1: %ld ms, Part 2: %ld ms\n",
         chrono::duration_cast<chrono::milliseconds>(end1 - start).count(),
         chrono::duration_cast<chrono::milliseconds>(end - end1).count());
  printf("Total time: %ld ms\n",
         chrono::duration_cast<chrono::milliseconds>(end - start).count());

  for (int i = 0; i < 3; ++i) {
    cudaMemcpy(output[i], output_gpu[i], nrow * ncol * sizeof(double),
        cudaMemcpyDeviceToHost);
  }

  saveImage(output, "test.png", nrow, ncol);

  // Memory leak? Who cares!
  return 0;
}