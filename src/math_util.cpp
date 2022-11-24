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
  for (int i = 0; i < nrow - 1; i++) {
    for (int j = 0; j < ncol; j++) {
      gradx[ncol * i + j] = input[ncol * (i + 1) + j] - input[ncol * i + j];
    }
  }

  for (int j = 0; j < ncol; j++) {
    gradx[ncol * (nrow - 1) + j] = -1 * input[ncol * (nrow - 1) + j];
  }

  // grady
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
  for (int j = 0; j < ncol; j++) {
    div[j] = gradx[j];
  }
  for (int i = 1; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      div[ncol * i + j] = gradx[ncol * i + j] - gradx[ncol * (i - 1) + j];
    }
  }

  // Then add grady to div
  for (int i = 0; i < nrow; i++) {
    // First column.
    div[ncol * i] += grady[ncol * i];

    for (int j = 1; j < nrow; j++) {
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
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] + b[i];
  }
}

void element_subtract(double* a, double* b, double* result, int nrow,
                      int ncol) {
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] - b[i];
  }
}

void element_multiply(double* a, double* b, double* result, int nrow,
                      int ncol) {
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] * b[i];
  }
}

void element_scale(double* a, double b, double* result, int nrow,
                      int ncol) {
  for (int i = 0; i < nrow * ncol; i++) {
    result[i] = a[i] * b;
  }
}

double array_sum(double* a, int nrow, int ncol) {
  double result = 0;
  for (int i = 0; i < nrow * ncol; i++) {
    result += a[i];
  }

  return result;
}