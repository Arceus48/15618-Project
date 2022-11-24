#include <math.h>
#include <stdio.h>

#include "data_util.cpp"
#include "math_util.cpp"

const int EDGE_WIDTH = 5;

/**
 * Allocate space for output before calling the function!
 *
 * @param[in] D shape: nrow * ncol
 * @param[in] input shape: nrow * ncol
 * @param[in] output shape: nrow * ncol
 * @param nrow
 * @param ncol
 * @param conv
 * @param niter
 */
void gradIntegrate(double* D, double* input, double* output, int nrow, int ncol,
                   double conv = 1e-3, int niter = 2000) {
  // With a parenthesis, it initialize all elements to be 0.
  double* B = new double[nrow * ncol]();
  double* r = new double[nrow * ncol];
  double* tmp = new double[nrow * ncol];
  double* q = new double[nrow * ncol];

  for (int i = EDGE_WIDTH; i < nrow - EDGE_WIDTH; i++) {
    for (int j = EDGE_WIDTH; j < ncol - EDGE_WIDTH; j++) {
      B[i * ncol + j] = 1.0;
    }
  }

  // Allocate space for output before calling the function!
  output = input;

  laplacian(output, r, nrow, ncol);
  element_subtract(D, r, r, nrow, ncol);
  element_multiply(B, r, r, nrow, ncol);

  double* d = r;

  element_multiply(r, r, tmp, nrow, ncol);
  double delta = array_sum(tmp, nrow, ncol);

  double denom;
  double ita;
  double delta_old;
  double beta;
  for (int i = 0; i < niter; i++) {
    double loss = sqrt(delta);
    if (loss < conv) {
      break;
    }

    printf("Iter: %d, loss: %f", i, loss);

    laplacian(d, q, nrow, ncol);
    element_multiply(d, q, tmp, nrow, ncol);
    denom = array_sum(tmp, nrow, ncol);
    ita = delta / denom;

    element_multiply(B, d, tmp, nrow, ncol);
    element_scale(tmp, ita, tmp, nrow, ncol);
    element_add(output, tmp, output, nrow, ncol);

    element_scale(q, ita, tmp, nrow, ncol);
    element_subtract(r, tmp, tmp, nrow, ncol);
    element_multiply(B, tmp, r, nrow, ncol);

    delta_old = delta;

    element_multiply(r, r, tmp, nrow, ncol);
    delta = array_sum(tmp, nrow, ncol);

    beta = delta / delta_old;
    element_scale(d, beta, tmp, nrow, ncol);
    element_add(r, tmp, d, nrow, ncol);
  }

  delete[] B;
  delete[] r;
  delete[] tmp;
  delete[] q;
}