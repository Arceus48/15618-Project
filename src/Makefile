CC = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
INCLUDES = -I/usr/local/include/opencv4
LDFLAGS = -L/usr/local/lib
LDLIBS = -lopencv_imgcodecs -lopencv_core

all: gradient_domain

gradient_domain: data_util.cpp math_util.cpp gradient_domain.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

clean:
	rm *.o gradient_domain