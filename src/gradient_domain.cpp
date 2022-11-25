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
  gradient(A, gradA, gradA + nrow * ncol, nrow, ncol);
  gradient(F, gradF, gradF + nrow * ncol, nrow, ncol);

  double* tmp1 = new double[nrow * ncol * 2];
  double* tmp2 = new double[nrow * ncol];
  double* tmp3 = new double[nrow * ncol];
  double* nume = new double[nrow * ncol];
  double* denom = new double[nrow * ncol];
  double* M = new double[nrow * ncol];
  double* Ws = new double[nrow * ncol];
  double* one_M = new double[nrow * ncol];
  double* one_Ws = new double[nrow * ncol];

  element_multiply(gradA, gradF, tmp1, nrow, ncol * 2);
  element_add(tmp1, tmp1 + nrow * ncol, tmp2, nrow, ncol);
  element_abs(tmp2, nume, nrow, ncol);

  element_multiply(gradA, gradA, tmp1, nrow, ncol * 2);
  element_add(tmp1, tmp1 + nrow * ncol, tmp2, nrow, ncol);
  element_multiply(gradF, gradF, tmp1, nrow, ncol * 2);
  element_add(tmp1, tmp1 + nrow * ncol, tmp3, nrow, ncol);
  element_multiply(tmp1, tmp2, denom, nrow, ncol);
  element_sqrt(denom, denom, nrow, ncol);

  // tmp2 holds magA, tmp3 holds magF
  element_divide_skip_0(nume, denom, M, nrow, ncol, 0);
  element_set_value_below_threshold(M, tmp2, nrow, ncol, 5e-3, 1.0);
  element_set_value_below_threshold(M, tmp3, nrow, ncol, 5e-3, 0.0);

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

  delete[] tmp1;
  delete[] tmp2;
  delete[] tmp3;
  delete[] nume;
  delete[] denom;
  delete[] M;
  delete[] Ws;
  delete[] one_M;
  delete[] one_Ws;
}

int main(int argc, char** argv) { return 0; }