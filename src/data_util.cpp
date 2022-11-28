#include "data_util.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace std;

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
double** readImage(const char* file_name, int* rowNum, int* colNum) {
  Mat image = imread(file_name);
  double** result = new double*[3];
  for (int i = 0; i < 3; i++) {
    result[i] = new double[image.rows * image.cols * 3];
  }
  unsigned char* input = (unsigned char*)(image.data);
  for (int i = 0; i < image.rows * image.cols; i++) {
    result[0][i] = input[3 * i] / 256.0;
    result[1][i] = input[3 * i + 1] / 256.0;
    result[2][i] = input[3 * i + 2] / 256.0;
  }

  *rowNum = image.rows;
  *colNum = image.cols;
  return result;
}

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
void saveImage(double** data, const char* file_name, int rowNum, int colNum) {
  Mat image = Mat(rowNum, colNum, CV_8UC3);
  unsigned char* output = (unsigned char*)(image.data);
  for (int i = 0; i < rowNum * colNum; i++) {
    output[3 * i] = std::min(255, std::max(0, (int)(data[0][i] * 256)));
    output[3 * i + 1] = std::min(255, std::max(0, (int)(data[1][i] * 256)));
    output[3 * i + 2] = std::min(255, std::max(0, (int)(data[2][i] * 256)));
  }

  imwrite(file_name, image);
}
