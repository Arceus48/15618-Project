### Install OpenCV
1. Follow the link: [Install on Linux](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
2. I used WSL to run the code. Not sure if it works on Windows...

### How to use it

```bash
./src/gradient_domain ./img/museum/museum_ambient.png ./img/museum/museum_flash.png 10 0.5 2 2 0.05 500
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
