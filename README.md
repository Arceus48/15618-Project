### Demo Video
https://drive.google.com/file/d/1iBacXAEyxzsSgcf4xaauOSkMr72gTlk7/view?usp=sharing


### Run GPU Binary Directly
```bash
./src/gradient_domain_gpu ./img/museum/museum_ambient.png ./img/museum/museum_flash.png 10 0.5 2 2 0.005 1000
```

### Use Python wrapper for Video Poisson patching
You may need to specify loss and niter accordingly.
```python
from interactive_wrapper.py import *
video_poisson_fusion('demo_video.mp4', '/tmp/patch.png', 50, 300)
```

### Install OpenCV
1. Run the commands at your home directory below to install OpenCV

```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
mkdir -p build && cd build
# Not sure if $HOME should work.. If not, use your absolute path here
cmake  ../opencv-4.x -DCMAKE_INSTALL_PREFIX=$HOME/opencv_install
make install
```
2. Add the following line to your bashrc:

```bash
# The cuda path was added previously...
export LD_LIBRARY_PATH=$HOME/opencv_install/lib:/usr/local/cuda-11.7/lib64/:${LD_LIBRARY_PATH}
```

### Install Python Dependency
`pip3 install -U pims MoviePy scikit-image`

### How to use it

```bash
./src/gradient_domain ./img/museum/museum_ambient.png ./img/museum/museum_flash.png 10 0.5 2 2 0.005 1000
```

Arguments:
1. Path to ambient image
2. Path to flash image
3. sig
4. threshold
5. bound_cond
6. init_opt
7. convergence check number
8. max iteration

### OpenMP
Set number of threads:
```bash
export OMP_NUM_THREADS=<num>
```

### Profiling
Get cache misses:
```bash
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses ./src/gradient_domain ./img/museum/museum_ambient.png ./img/museum/museum_flash.png 10 0.5 2 2 0.005 1000
```

Time functions:
```bash
perf record <executable>

perf report
```
