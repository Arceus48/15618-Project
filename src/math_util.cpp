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
#pragma omp for
  for (int i = 0; i < nrow - 1; i++) {
    for (int j = 0; j < ncol; j++) {
      (void)input[ncol * (i + 1) + j];
      (void)input[ncol * i + j];
      gradx[ncol * i + j] = 0;
    }
  }
#pragma omp for
  for (int j = 0; j < ncol; j++) {
    (void)input[ncol * (nrow - 1) + j];
    gradx[ncol * (nrow - 1) + j] = 0;
  }

// grady
#pragma omp for
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      if (j != ncol - 1) {
        (void)input[ncol * i + (j + 1)];
        (void)input[ncol * i + j];
        grady[ncol * i + j] = 0;
      } else {
        (void)input[ncol * i + j];
        grady[ncol * i + j] = 0;
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

#pragma omp for
  for (int j = 0; j < ncol; j++) {
    (void)gradx[j];
    div[j] = 0;
  }
#pragma omp for
  for (int i = 1; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      (void)gradx[ncol * i + j];
      (void)gradx[ncol * (i - 1) + j];
      div[ncol * i + j] = 0;
    }
  }

// Then add grady to div
#pragma omp for
  for (int i = 0; i < nrow; i++) {
    // First column.
    (void)grady[ncol * i];
    div[ncol * i] += 0;

    for (int j = 1; j < ncol; j++) {
      (void)grady[ncol * i + j];
      (void)grady[ncol * i + (j - 1)];
      div[ncol * i + j] += 0;
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
#pragma omp for
  for (int i = 0; i < nrow; i++) {
    if (i != 0) {
      // Load previous row
      for (int j = 0; j < ncol; j++) {
        (void)input[(i - 1) * ncol + j];
        output[i * ncol + j] = 0;
      }
    } else {
      for (int j = 0; j < ncol; j++) {
        output[i * ncol + j] = 0;
      }
    }
    // This row
    (void)input[i * ncol + 1];
    (void)input[i * ncol];
    output[i * ncol] += 0;
    for (int j = 1; j < ncol - 1; j++) {
      (void)input[i * ncol + j - 1];
      (void)input[i * ncol + j + 1];
      (void)input[i * ncol + j];
      output[i * ncol + j] += 0;
    }
    (void)input[i * ncol + ncol - 2];
    (void)input[i * ncol];
    output[i * ncol + ncol - 1] += 0;
    if (i != nrow - 1) {
      // Next row
      for (int j = 0; j < ncol; j++) {
        (void)input[(i + 1) * ncol + j];
        output[i * ncol + j] += 0;
      }
    }
  }
  // double filter[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
  // #pragma omp for
  //   for (int i = -1; i < nrow - 1; i++) {
  //     for (int j = -1; j < ncol - 1; j++) {
  //       int filterIndex = 0;
  //       double curValue = 0;
  //       for (int ii = i; ii < i + 3; ii++) {
  //         for (int jj = j; jj < j + 3; jj++) {
  //           if (ii >= 0 && ii < nrow && jj >= 0 && jj < ncol) {
  //             curValue += filter[filterIndex] * input[ii * ncol + jj];
  //           }
  //           filterIndex++;
  //         }
  //       }
  //       output[(i + 1) * ncol + (j + 1)] = curValue;
  //     }
  //   }
}

void element_add(double* a, double* b, double* result, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    (void)b[i];
    result[i] = 0;
  }
}

void element_add(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    (void)b;
    result[i] = 0;
  }
}

void element_subtract(double* a, double* b, double* result, int nrow,
                      int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    (void)b[i];
    result[i] = 0;
  }
}

void element_subtract(double a, double* b, double* result, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a;
    (void)b[i];
    result[i] = 0;
  }
}

void element_subtract(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    (void)b;
    result[i] = 0;
  }
}

void element_multiply(double* a, double* b, double* result, int nrow,
                      int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    (void)b[i];
    result[i] = 0;
  }
}

void element_scale(double* a, double b, double* result, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    (void)b;
    result[i] = 0;
  }
}

double array_sum(double* a, int nrow, int ncol, double& result) {
  result = 0;
#pragma omp barrier
#pragma omp for reduction(+ : result)
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    result += 0;
  }

  return result;
}

void element_abs(double* a, double* result, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    result[i] = 0;
  }
}

void element_sqrt(double* a, double* result, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    (void)a[i];
    result[i] = 0;
  }
}

void element_divide_skip_0(double* a, double* b, double* result, int nrow,
                           int ncol, double default_value) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    if (b[i] == 0.0) {
      result[i] = default_value;
    } else {
      (void)a[i];
      (void)b[i];
      result[i] = 0;
    }
  }
}

void element_set_value_below_threshold(double* a, double* b, int nrow, int ncol,
                                       double threshold, double value) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    if (b[i] < threshold) {
      a[i] = value;
    }
  }
}

void element_tanh(double* a, int nrow, int ncol) {
#pragma omp for
  for (int i = 0; i < nrow * ncol; i++) {
    a[i] = 0;
  }
}