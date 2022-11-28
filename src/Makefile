CC = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
INCLUDES = -I/usr/local/include/opencv4
LDFLAGS = -L/usr/local/lib
LDLIBS = -lopencv_imgcodecs -lopencv_core

all: gradient_domain

gradient_domain: data_util.o math_util.o gradient_domain.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

gradient_domain.o: gradient_domain.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

data_util.o: data_util.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

math_util.o: math_util.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

.PHONY: clean

clean:
	rm *.o gradient_domain