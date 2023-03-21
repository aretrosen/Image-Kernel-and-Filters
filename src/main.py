from pathlib import Path

import numpy as np
from PIL import Image

from helper import add_noise
from helper import convolve2d
from helper import gaussian_filtering
from helper import harris
from helper import histogram_equalization
from helper import LoG_kernel
from helper import median_filtering
from helper import otsu
from helper import psnr
from helper import save_img
from helper import save_npy
from helper import legall, ilegall
from helper import zero_crossing

##########
# Part 1 #
##########
def load_image(path: Path) -> np.ndarray:
    """Loads Image.

    Args:
        path (Path): Path of the image.

    Returns:
        Array representation of the image.

    """
    try:
        img_arr: np.ndarray = np.load(npy_path.joinpath(path.stem + "_arr.npy"))
        return img_arr
    except:
        pass
    try:
        img_arr = np.array(Image.open(path))
        save_npy(img_arr, npy_path.joinpath(path.stem + "_arr.npy"))
        return img_arr
    except:
        raise FileNotFoundError

def rgb_to_gray(arr: np.ndarray) -> None:
    """Computes and saves grayscale image from a colorascale image's array

    Args:
        arr (np.ndarray): Array representation of the color image.
    """
    red: np.ndarray = arr[:, :, 0]
    green: np.ndarray = arr[:, :, 1]
    blue: np.ndarray = arr[:, :, 2]
    gray: np.ndarray = np.rint(0.2989 * red + 0.5870 * green + 0.1141 * blue)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    save_npy(gray, npy_path.joinpath(path.stem + "_grayscale.npy"))
    save_img(gray, img_path.joinpath(path.stem + "_grayscale.jpg"))

##########
# Part 2 #
##########
def adding_noise_and_denoising(im: Path) -> None:
    try:
        ia: np.ndarray = np.load(im)
    except:
        raise FileNotFoundError

    # Adding Noise with random gaussian noise of σ = 5.
    noisy: np.ndarray = add_noise(ia, 5)
    save_img(noisy, img_path.joinpath(path.stem + "_noisy.jpg"))

    # Wavelet filtering (Hard thresholding)
    thresholds: dict[str, float] = dict()
    thresholds["HH"] = float(
        input(
            "Enter threshold for HH in wavelet-filtering. This is the most "
            "significant, don't let it be default. (Default: 0.0): "
        )
        or 0
    )
    thresholds["LH"] = float(
        input("Enter threshold for LH in wavelet-filtering (Default: 0.0): ") or 0
    )
    thresholds["HL"] = float(
        input("Enter threshold for HL in wavelet-filtering (Default: 0.0): ") or 0
    )
    bands: dict[str, np.ndarray] = legall(noisy, thresholds)
    save_img(
        np.clip(np.rint(bands["LL"] * 2), 0, 255).astype(np.uint8),
        img_path.joinpath(path.stem + "_noisy_wavelet_LL.jpg"),
    )
    wf: np.ndarray = ilegall(bands)
    save_img(wf, img_path.joinpath(path.stem + "_noisy_wavelet_filtered.jpg"))

    # Median filtering
    mf: np.ndarray = median_filtering(noisy)
    save_img(mf, img_path.joinpath(path.stem + "_noisy_median_filtered.jpg"))

    # Gaussian Filtering with σ = 4
    gf: np.ndarray = gaussian_filtering(noisy, 4)
    save_img(gf, img_path.joinpath(path.stem + "_noisy_gaussian_filtered.jpg"))

    # Print all PSNRs
    print(f"PSNR of noisy image = {psnr(ia, noisy)}")
    print(f"PSNR of wavelet-filtered image = {psnr(ia, wf)}")
    print(f"PSNR of median-filtered image = {psnr(ia, mf)}")
    print(f"PSNR of gaussian-filtered image = {psnr(ia, gf)}")

##########
# Part 3 #
##########
def LoG_and_zero_crossing(im: Path, sigma: float, zth: float = 0) -> None:
    """Computes the edges of a grayscale image.

    Convolves an image with LoG kernel of given standard deviation σ, then
    computes the zero crossings, with a threshold provided (ideally 0).

    Args:
        im (Path): The path to the array representation of the grayscale image.
        sigma (float): The standard deviation σ of the LoG kernel.
        zth (float): Zero thresholding coefficient. It must be negative.
    """
    try:
        gray: np.ndarray = np.load(im)
    except:
        raise FileNotFoundError
    if zth > 0:
        raise ValueError(
            f"The value of zth is expected to be negative, but {zth} is provided."
        )
    try:
        res: np.ndarray = np.load(
            npy_path.joinpath(im.stem + f"_convolved_LoG_sigma_{sigma}.npy")
        )
    except:
        res = convolve2d(gray, LoG_kernel(sigma))
        save_npy(
            res,
            npy_path.joinpath(im.stem + f"_convolved_LoG_sigma_{sigma}.npy"),
        )
    save_img(
        zero_crossing(res, zth),
        img_path.joinpath(im.stem + f"_zero_crossing_LoG_{sigma}.jpg"),
    )
    res_min: int = np.amin(res)
    if res_min < 0:
        res = res - res_min
    res = 255 * res / np.amax(res)
    save_img(res.astype(np.uint8), img_path.joinpath(im.stem + f"_LoG_{sigma}.jpg"))

##########
# Part 4 #
##########
def equalized_grayscale(file: Path) -> None:
    """Adjusts contrast of an image.

    Args:
        file (Path): The path to the array representation of the grayscale image.

    """
    try:
        gray: np.ndarray = np.load(file)
    except:
        raise FileNotFoundError
    assert len(gray.shape) == 2
    equalized: np.ndarray = histogram_equalization(gray)
    save_npy(equalized, npy_path.joinpath(file.stem + "_equalized.npy"))
    save_img(equalized, img_path.joinpath(file.stem + "_equalized.jpg"))

def otsu_binarization(file: Path) -> None:
    """Automatic binarization of grayscale image using Otsu's method.

    Args:
        file (Path): The path to the array representation of a grayscale image.
    """
    try:
        gray: np.ndarray = np.load(file)
    except:
        raise FileNotFoundError
    ots_th: np.ndarray = otsu(gray)
    save_img(ots_th, img_path.joinpath(file.stem + "_otsu.jpg"))

##########
# Part 5 #
##########
def harris_corner_detector(
    file: Path,
    k: float = 0.05,
    sigma_1: float = 1,
    sigma_2: float = 3,
    threshold: float = 1,
) -> None:
    """Finds Corners in a grayscale image using Harris's method.

    Args:
        file (Path): Path of the array representation of a grayscale image.
        k (float): Harris coefficient. k ⋴ [0.04, 0.06] generally.
        sigma_1 (float) : Standar deviation σ for Gaussian image smoothing.
        sigma_2 (float) : Standar deviation σ for Gaussian Autocorrelation
          coefficient smoothing.
        threshold: The threshold above which all R values are considered as
          maximum. Others are ignored.
    """
    try:
        gray: np.ndarray = np.load(file)
    except:
        raise FileNotFoundError
    cornered = harris(gray, k, sigma_1, sigma_2, threshold)
    save_img(cornered, img_path.joinpath(file.stem + "_corners.jpg"))

if __name__ == "__main__":
    path: Path = Path("../data/external/Butterfly.JPG").resolve()
    npy_path: Path = Path("../data/interim").resolve()
    img_path: Path = Path("../data/generated_images").resolve()
    img_arr = load_image(path)
    rgb_to_gray(img_arr)
    LoG_and_zero_crossing(npy_path.joinpath("Butterfly_grayscale.npy"), 5, -0.05)
    adding_noise_and_denoising(npy_path.joinpath("Butterfly_arr.npy"))
    equalized_grayscale(npy_path.joinpath("Butterfly_grayscale.npy"))
    otsu_binarization(npy_path.joinpath("Butterfly_grayscale.npy"))
    otsu_binarization(npy_path.joinpath("Butterfly_grayscale_equalized.npy"))
    harris_corner_detector(
        npy_path.joinpath("Butterfly_grayscale.npy"),
        sigma_1=3,
        threshold=2,
    )
