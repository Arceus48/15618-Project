#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <iostream>

#include "data_util.h"
#include "math_util.h"
using namespace std;

const int EDGE_WIDTH = 5;

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
#pragma omp for nowait collapse(2)
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
  element_subtract(D, r, r, nrow, ncol);
#pragma omp barrier
  element_multiply(B, r, r, nrow, ncol);
#pragma omp barrier

#pragma omp for nowait
  // Allocate space for output before calling the function!
  for (int i = 0; i < nrow * ncol; i++) {
    d[i] = r[i];
  }

  element_multiply(r, r, tmp, nrow, ncol);
#pragma omp barrier
  double delta = array_sum(tmp, nrow, ncol, result);
  double denom;
  double ita;
  double delta_old;
  double beta;
  for (int i = 0; i < niter; i++) {
#pragma omp barrier
    double loss = sqrt(delta);
#pragma omp single
    printf("Iter: %d, loss: %f\n", i, loss);
    if (loss <= conv) {
      break;
    }

    laplacian(d, q, nrow, ncol);
    element_multiply(d, q, tmp, nrow, ncol);
#pragma omp barrier
    denom = array_sum(tmp, nrow, ncol, result);
    ita = delta / denom;
#pragma omp barrier
    element_multiply(B, d, tmp, nrow, ncol);
#pragma omp barrier
    element_scale(tmp, ita, tmp, nrow, ncol);
#pragma omp barrier
    element_add(output, tmp, output, nrow, ncol);
#pragma omp barrier
    element_scale(q, ita, tmp, nrow, ncol);
#pragma omp barrier
    element_subtract(r, tmp, tmp, nrow, ncol);
#pragma omp barrier
    element_multiply(B, tmp, r, nrow, ncol);
#pragma omp barrier
    delta_old = delta;
    element_multiply(r, r, tmp, nrow, ncol);
#pragma omp barrier
    delta = array_sum(tmp, nrow, ncol, result);
#pragma omp barrier
    beta = delta / delta_old;
#pragma omp barrier
    element_scale(d, beta, tmp, nrow, ncol);
#pragma omp barrier
    element_add(r, tmp, d, nrow, ncol);
#pragma omp barrier
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
void fuseGrad(double* A, double* F, double sig, double thld, double* gradA,
              double* gradF, double* fused, int nrow, int ncol) {
#pragma omp parallel
  {
    gradient(A, gradA, gradA + nrow * ncol, nrow, ncol);
    gradient(F, gradF, gradF + nrow * ncol, nrow, ncol);
#pragma omp barrier
    element_multiply(gradA, gradF, tmp1, nrow, ncol * 2);
#pragma omp barrier
    element_add(tmp1, tmp1 + nrow * ncol, tmp2, nrow, ncol);
#pragma omp barrier
    element_abs(tmp2, nume, nrow, ncol);

    element_multiply(gradA, gradA, tmp1, nrow, ncol * 2);
#pragma omp barrier
    element_add(tmp1, tmp1 + nrow * ncol, tmp2, nrow, ncol);
#pragma omp barrier
    element_multiply(gradF, gradF, tmp1, nrow, ncol * 2);
#pragma omp barrier
    element_add(tmp1, tmp1 + nrow * ncol, tmp3, nrow, ncol);
#pragma omp barrier
    element_multiply(tmp2, tmp3, denom, nrow, ncol);
#pragma omp barrier
    element_sqrt(denom, denom, nrow, ncol);
#pragma omp barrier
    // tmp2 holds magA ^ 2, tmp3 holds magF ^ 2
    element_divide_skip_0(nume, denom, M, nrow, ncol, 0);
#pragma omp barrier
    element_set_value_below_threshold(M, tmp2, nrow, ncol, 5e-3 * 5e-3, 1.0);
#pragma omp barrier
    element_set_value_below_threshold(M, tmp3, nrow, ncol, 5e-3 * 5e-3, 0.0);
#pragma omp barrier
    element_subtract(F, thld, tmp2, nrow, ncol);
#pragma omp barrier
    element_scale(tmp2, sig, Ws, nrow, ncol);
#pragma omp barrier
    element_tanh(Ws, nrow, ncol);
#pragma omp barrier
    element_add(Ws, 1.0, Ws, nrow, ncol);
#pragma omp barrier
    element_scale(Ws, 0.5, Ws, nrow, ncol);
#pragma omp barrier
    element_subtract(1.0, Ws, one_Ws, nrow, ncol);
    element_subtract(1.0, M, one_M, nrow, ncol);
#pragma omp barrier
    // Merge into fuse. Done half by half to avoid stacking.
    element_multiply(one_M, gradA, fused, nrow, ncol);
    element_multiply(M, gradF, tmp2, nrow, ncol);
#pragma omp barrier
    element_add(tmp2, fused, fused, nrow, ncol);
#pragma omp barrier
    element_multiply(fused, one_Ws, fused, nrow, ncol);
    element_multiply(Ws, gradA, tmp3, nrow, ncol);
#pragma omp barrier
    element_add(fused, tmp3, fused, nrow, ncol);
    element_multiply(one_M, gradA + nrow * ncol, fused + nrow * ncol, nrow,
                     ncol);
    element_multiply(M, gradF + nrow * ncol, tmp2, nrow, ncol);
#pragma omp barrier
    element_add(tmp2, fused + nrow * ncol, fused + nrow * ncol, nrow, ncol);
#pragma omp barrier
    element_multiply(fused + nrow * ncol, one_Ws, fused + nrow * ncol, nrow,
                     ncol);
#pragma omp barrier
    element_multiply(Ws, gradA + nrow * ncol, tmp3, nrow, ncol);
#pragma omp barrier
    element_add(fused + nrow * ncol, tmp3, fused + nrow * ncol, nrow, ncol);
#pragma omp barrier
  }
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
  for (int i = 2; i >= 0; i--) {
    fuseGrad(A[i], F[i], sig, thld, gradA + i * 2 * nrow * ncol,
             gradF + i * 2 * nrow * ncol, fused + i * 2 * nrow * ncol, nrow,
             ncol);
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
#pragma omp parallel default(shared)
    {
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