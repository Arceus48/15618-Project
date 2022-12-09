#include "math_util.h"

#include <omp.h>
#include <stdio.h>

/**
 * Allocate memory for gradx and grady before calling the function!
 *
 * Not sure if "gradx" holds gradient for x. May got it wrong to hold y.
 *
 * @param[in] input size: nrow * ncol
 * @param[out] gradx size: nrow * ncol, corresponds to: grad[:, :, 0]
 * @param[out] grady size: nrow * ncol, corresponds to: grad[:, :, 1]
 * @param[in] nrow
 * @param[in] ncol
 */
void gradient(double* input, double* gradx, double* grady, int nrow, int ncol) {
// gradx
#pragma omp for nowait collapse(2)
  for (int i = 0; i < nrow - 1; i++) {
    for (int j = 0; j < ncol; j++) {
      gradx[ncol * i + j] = input[ncol * (i + 1) + j] - input[ncol * i + j];
    }
  }
#pragma omp for nowait
  for (int j = 0; j < ncol; j++) {
    gradx[ncol * (nrow - 1) + j] = -1 * input[ncol * (nrow - 1) + j];
  }

// grady
#pragma omp for collapse(2)
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      if (j != ncol - 1) {
        grady[ncol * i + j] = input[ncol * i + (j + 1)] - input[ncol * i + j];

      } else {
        grady[ncol * i + j] = -1 * input[ncol * i + j];
      }
    }
  }
}

/**
 * Allocate memory for div before calling the function!
 *
 * Modify it to be nrow * ncol
 *
 * @param[in] gradx input_vec[:,:,0], shape: nrow * ncol
 * @param[in] grady input_vec[:,:,1], shape: nrow * ncol
 * @param[out] div shape: nrow * ncol
 * @param[in] nrow
 * @param[in] ncol
 */
void divergence(double* gradx, double* grady, double* div, int nrow, int ncol) {
  // First assign gradx
  // For first line, directly copy the value

#pragma omp for nowait
  for (int j = 0; j < ncol; j++) {
    div[j] = gradx[j];
  }
#pragma omp for nowait
  for (int i = 1; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      div[ncol * i + j] = gradx[ncol * i + j] - gradx[ncol * (i - 1) + j];
    }
  }
#pragma omp barrier
// Then add grady to div
#pragma omp for
  for (int i = 0; i < nrow; i++) {
    // First column.
    div[ncol * i] += grady[ncol * i];

    for (int j = 1; j < ncol; j++) {
      div[ncol * i + j] += grady[ncol * i + j] - grady[ncol * i + (j - 1)];
    }
  }
}

/**
 * Allocate memory for output before calling this function!
 *
 * @param[in] input shape: nrow * ncol
 * @param[in] output shape: nrow * ncol
 * @param[in] nrow
 * @param[in] ncol
 */
void laplacian(double* input, double* output, int nrow, int ncol) {
  // Since the filter is symmetric with respect to a point, no need to mirror
  // it.
  double filter[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
#pragma omp for collapse(2)
  for (int i = -1; i < nrow - 1; i++) {
    for (int j = -1; j < ncol - 1; j++) {
      int filterIndex = 0;
      double curValue = 0;
      for (int ii = i; ii < i + 3; ii++) {
        for (int jj = j; jj < j + 3; jj++) {
          if (ii >= 0 && ii < nrow && jj >= 0 && jj < ncol) {
            curValue += filter[filterIndex] * input[ii * ncol + jj];
          }
          filterIndex++;
        }
      }
      output[(i + 1) * ncol + (j + 1)] = curValue;
    }
  }
}

void element_add(double* a, double* b, double* result, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] + b[i];
  }
}

void element_add(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] + b;
  }
}

void element_subtract(double* a, double* b, double* result, int nrow,
                      int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] - b[i];
  }
}

void element_subtract(double a, double* b, double* result, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a - b[i];
  }
}

void element_subtract(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] - b;
  }
}

void element_multiply(double* a, double* b, double* result, int nrow,
                      int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] * b[i];
  }
}

void element_scale(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] * b;
  }
}

double array_sum(double* a, int nrow, int ncol, double& result) {
  result = 0;
#pragma omp for reduction(+ : result)
  for (int i = 0; i < nrow * ncol; i++) {
    result += a[i];
  }

  return result;
}

void element_abs(double* a, double* result, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = fabs(a[i]);
  }
}

void element_sqrt(double* a, double* result, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = sqrt(a[i]);
  }
}

void element_divide_skip_0(double* a, double* b, double* result, int nrow,
                           int ncol, double default_value) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    if (b[i] == 0.0) {
      result[i] = default_value;
    } else {
      result[i] = a[i] / b[i];
    }
  }
}

void element_set_value_below_threshold(double* a, double* b, int nrow, int ncol,
                                       double threshold, double value) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    if (b[i] < threshold) {
      a[i] = value;
    }
  }
}

void element_tanh(double* a, int nrow, int ncol) {
#pragma omp for nowait
  for (int i = 0; i < nrow * ncol; i++) {
    a[i] = tanh(a[i]);
  }
}