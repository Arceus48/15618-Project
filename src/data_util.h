/**
 * Read the image and return in three double array with size: row * col.
 * The color channel is BGR, not RGB!
 *
 * @param[in] file_name input name
 * @param[out] rowNum output
 * @param[out] colNum output
 * @return double*[3]. Each pointer points to an array with shape row * col. The
 * channel order is BGR.
 */
double** readImage(const char* file_name, int* rowNum, int* colNum);

/**
 * Store the data (row * col * 3, BGR color order) within range [0, 1] to the
 * output file.
 *
 * @param[in] data: double*[3]. Each pointer points to an array with shape row *
 * col. Three arrays in the order BGR.
 * @param[in] file_name
 * @param[in] rowNum
 * @param[in] colNum
 */
void saveImage(double** data, const char* file_name, int rowNum, int colNum);