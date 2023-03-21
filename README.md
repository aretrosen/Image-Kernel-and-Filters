# ADIP & CV (CS60052) Assignment 1

# Installation:
To run the project, you will need to download and install Miniconda/Anaconda.
Follow this link for [Miniconda](https://docs.conda.io/en/latest/miniconda.html "Miniconda Page"). Then, to install required packages, activate `conda` environment,and create environment by running 

	conda env create -f environment.yml. 

I personally recommend using mamba, as it's faster. Also, this project heavily relies on numba JIT compilation, and is required. You may remove numba from the source code and remove all @njit decorators, but do it at your own risk. This simple njit decorators reduce the running time drastically by some order.
The program runs in my laptop in less than 2 minutes, including the time for input. Thus, this makes it a lot faster than usual. Then go to src directory,
and run

	python main.py

All functions are heavily documented, and all images generated after running main.py are kept in the data/generated_images folder. `PSNRs` will be directed to standard output only, and not stored in a file. The inputs and results, as in my computer is documented below:

- Enter threshold for HH in wavelet-filtering. This is the most significant, don't let it be default. (Default: 0.0): 255
- Enter threshold for LH in wavelet-filtering (Default: 0.0): 
- Enter threshold for HL in wavelet-filtering (Default: 0.0): 
- PSNR of noisy image = 34.142559411321876
- PSNR of wavelet-filtered image = 35.305220451052286
- PSNR of median-filtered image = 40.696856904986426
- PSNR of gaussian-filtered image = 37.143623215884666

As we see, median filtering is most successful here. Wavelet filtering could be more, but I haven't tested all combination for thresholding. The wavelet is otherwise completely lossless, because if all defaults to 0, PSNR of it is same as that of noisy image, or the input image.

For setting other thresholds, you can comment out specific parts of the code and run it with different thresholds. I personally recommend changing the threshold for zero-crossing, and Harris corner detection. Experiment with it a bit, I have only checked with a few values, and kept whichever I liked most. Explanation and rationale of all choices are documented as docstrings, and checking them out will help in understanding the use of each function.


### Author Information:
- **Name** : Aritra Sen.
- **Roll no.** : 19ME10101.
- **Department** : Mechanical Engineering.
- **Subject** : Advanced Digital Image Processing & Computer Vision (CS60052).
