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
#pragma omp parallel default(shared)
  {
#pragma omp for
    for (int i = 0; i < nrow - 1; i++) {
      for (int j = 0; j < ncol; j++) {
        gradx[ncol * i + j] = input[ncol * (i + 1) + j] - input[ncol * i + j];
      }
    }
#pragma omp for
    for (int j = 0; j < ncol; j++) {
      gradx[ncol * (nrow - 1) + j] = -1 * input[ncol * (nrow - 1) + j];
    }

// grady
#pragma omp for
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
#pragma omp parallel default(shared)
  {
#pragma omp for
    for (int j = 0; j < ncol; j++) {
      div[j] = gradx[j];
    }
#pragma omp for
    for (int i = 1; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        div[ncol * i + j] = gradx[ncol * i + j] - gradx[ncol * (i - 1) + j];
      }
    }

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
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow; i++) {
    if (i != 0) {
      // Load previous row
      for (int j = 0; j < ncol; j++) {
        output[i * ncol + j] = input[(i - 1) * ncol + j];
      }
    } else {
      for (int j = 0; j < ncol; j++) {
        output[i * ncol + j] = 0;
      }
    }
    // This row
    output[i * ncol] += input[i * ncol + 1] - 4 * input[i * ncol];
    for (int j = 1; j < ncol - 1; j++) {
      output[i * ncol + j] += input[i * ncol + j - 1] + input[i * ncol + j + 1] - 4 * input[i * ncol + j];
    }
    output[i * ncol + ncol - 1] += input[i * ncol + ncol - 2] - 4 * input[i * ncol];
    if (i != nrow - 1) {
      // Next row
      for (int j = 0; j < ncol; j++) {
        output[i * ncol + j] += input[(i + 1) * ncol + j];
      }
    }
  }
}

void element_add(double* a, double* b, double* result, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] + b[i];
  }
}

void element_add(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] + b;
  }
}

void element_subtract(double* a, double* b, double* result, int nrow,
                      int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] - b[i];
  }
}

void element_subtract(double a, double* b, double* result, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a - b[i];
  }
}

void element_subtract(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] - b;
  }
}

void element_multiply(double* a, double* b, double* result, int nrow,
                      int ncol) {
#pragma omp parallel for
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] * b[i];
  }
}

void element_scale(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] * b;
  }
}

double array_sum(double* a, int nrow, int ncol) {
  double result = 0;
#pragma omp parallel for reduction(+ : result) default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result += a[i];
  }

  return result;
}

void element_abs(double* a, double* result, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = fabs(a[i]);
  }
}

void element_sqrt(double* a, double* result, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = sqrt(a[i]);
  }
}

void element_divide_skip_0(double* a, double* b, double* result, int nrow,
                           int ncol, double default_value) {
#pragma omp parallel for default(shared)
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
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    if (b[i] < threshold) {
      a[i] = value;
    }
  }
}

void element_tanh(double* a, int nrow, int ncol) {
#pragma omp parallel for default(shared)
  for (int i = 0; i < nrow * ncol; i++) {
    a[i] = tanh(a[i]);
  }
}