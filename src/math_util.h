#include <math.h>

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
void gradient(double* input, double* gradx, double* grady, int nrow, int ncol);

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
void divergence(double* gradx, double* grady, double* div, int nrow, int ncol);

/**
 * Allocate memory for output before calling this function!
 *
 * @param[in] input shape: nrow * ncol
 * @param[in] output shape: nrow * ncol
 * @param[in] nrow
 * @param[in] ncol
 */
void laplacian(double* input, double* output, int nrow, int ncol);

void element_add(double* a, double* b, double* result, int nrow, int ncol);

void element_add(double* a, double b, double* result, int nrow, int ncol);

void element_subtract(double* a, double* b, double* result, int nrow, int ncol);

void element_subtract(double a, double* b, double* result, int nrow, int ncol);

void element_subtract(double* a, double b, double* result, int nrow, int ncol);

void element_multiply(double* a, double* b, double* result, int nrow, int ncol);

void element_scale(double* a, double b, double* result, int nrow, int ncol);

double array_sum(double* a, int nrow, int ncol);

void element_abs(double* a, double* result, int nrow, int ncol);

void element_sqrt(double* a, double* result, int nrow, int ncol);

void element_divide_skip_0(double* a, double* b, double* result, int nrow,
                           int ncol, double default_value);

void element_set_value_below_threshold(double* a, double* b, int nrow, int ncol,
                                       double threshold, double value);

void element_tanh(double* a, int nrow, int ncol);