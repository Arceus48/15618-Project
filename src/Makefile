CC = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
INCLUDES = -I$$HOME/opencv_install/include/opencv4
LDFLAGS = -L$$HOME/opencv_install/lib -L/usr/local/cuda-11.7/lib64/
LDLIBS = -lopencv_imgcodecs -lopencv_core -fopenmp
LDLIBS_CUDA = -lopencv_imgcodecs -lopencv_core -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

######
# CU_FILES   := saxpy.cu
######

all: gradient_domain gradient_domain_gpu

gradient_domain: data_util.o math_util.o gradient_domain.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

gradient_domain_gpu: data_util.o gradient_domain_gpu.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS_CUDA)

gradient_domain.o: gradient_domain.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

gradient_domain_gpu.o: gradient_domain_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS_CUDA)

data_util.o: data_util.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

math_util.o: math_util.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

.PHONY: clean

clean:
	rm *.o gradient_domain gradient_domain_gpu