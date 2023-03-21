from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image

@njit
def LoG_kernel(sigma: float) -> np.ndarray:
    """Computes an approximate Laplacian of Gaussian Kernel.

    Given a standard deviation, σ, this program computes an approximate
    Laplacian of Gaussian Kernel, with a radius of about 3σ. This is done so
    because most of the area of the LOG function lies within an approximate
    radius of 3σ. This function assumes that the mean μ is 0.

    The values in the kernel are not exactly from the LoG function. They are
    scaled using a factor, so that sum of absolute values of all elements in the
    array is 1, and sum of all values of LoG kernel is 0, which is a property
    of LoG kernel. Rounding errors might creep in, so a mean value is subtracted
    from each value in the kernel.

    Args:
        sigma (float): The standard deviation σ.

    Returns:
        A kernel of 3σ radius, which is an approximation of a Laplacian of
        Gaussian function.

    """
    wbar: float = round(3 * sigma)
    w: float = 2 * wbar + 1
    tmp: np.ndarray = np.arange(w) - wbar
    x: np.ndarray = tmp.reshape(1, w)
    y: np.ndarray = tmp.reshape(w, 1)
    g: np.ndarray = np.square(x) + np.square(y)
    res: np.ndarray = np.multiply(
        (g - 2 * sigma * sigma), np.exp(-1 * g / (2 * sigma * sigma))
    )
    res = res - np.mean(res)
    return res / np.sum(np.abs(res))

@njit
def gaussian_kernel(sigma: float) -> np.ndarray:
    """Computes an approximate Gaussian kernel.

    This function computes an approximate Gaussian kernel with of given standard
    deviation σ, assuming a mean μ of 0. The radius is taken to be about 2.5σ
    since most of the area under the gaussian curve lies within this radius.

    This resulting gaussian kernelis scales so that the summation of all values
    equates to 1. This is thus a scaling to the original gaussian function, and
    are not exactly equivalent.

    Args:
        sigma (float): The standard deviation σ.

    Returns:
        A kernel which is an approximation of Gaussian function of given σ.

    """
    wbar: float = round(2.5 * sigma - 0.5)
    w: float = 2 * wbar + 1
    x: np.ndarray = np.square(np.arange(w) - wbar)
    x = np.exp(-1 * x / (2 * sigma * sigma))
    return x / np.sum(x)

@njit
def add_noise(im: np.ndarray, sigma: float) -> np.ndarray:
    """Adds Gaussian noise to an array (image).

    Adds random sample from Gaussian distribution of given standard deviation σ
    to each element of the array.

    Args:
        im (np.ndarray): The array representation of the image.
        sigma (float): The standard deviation of the Gaussian Distribution from
          which to sample.
    Returns:
        An array representation of a noisy image.
    """
    noisy: np.ndarray = im + np.random.normal(0, sigma, im.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)

@njit
def convolve2d_util(arr: np.ndarray, kern: np.ndarray, pad: int) -> np.ndarray:
    """Convolves a 2D array with an 2D kernel.

    This method is an util function for the actual convolve2d function. This
    function exists because numpy.pad is not supported by numba yet, and using
    numba speeds up this function considerably. More about numba in README.

    Args:
        arr (np.ndarray): 2D padded array to convolve.
        kern (np.ndarray): The 2D Kernel with which to convolve given array.
        pad (int): The size of padding in the initial input array.

    Returns:
        Convolution of given array with the given kernel.

    """
    h, w = arr.shape
    res: np.ndarray = np.zeros((h - pad + 1, w - pad + 1), dtype=np.float64)
    for y in range(h - pad + 1):
        for x in range(w - pad + 1):
            res[y][x] = np.sum(np.multiply(kern, arr[y : y + pad, x : x + pad]))
    return res

def convolve2d(arr: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """Returns the convolution of a 2D array with a 2D kernel.

    This method returns the result of convolution, after it pads the array
    provided. More about the rationale of this function in convolve2d_util.
    Note: This function is only needed when a 2D convolution with an unseparable
    kernel is required. Otherwise, use convolve2d_separable. The complexity of
    the convolve2d_separable function is lesser.

    Args:
        arr (np.ndarray): The array to convolve.
        kern (np.ndarray): The kernel with which to convolve.

    Returns:
        A convolution of the array with the kernel.
    """
    kern = np.flipud(np.fliplr(kern))
    s: int = kern.shape[0]
    arr = np.pad(arr, s // 2)
    return convolve2d_util(arr, kern, s)

@njit
def convolve2d_separable(
    arr: np.ndarray, hkern: np.ndarray, wkern: np.ndarray
) -> np.ndarray:
    """Convolves a 2D array with separable horizontal and vertical 1D kernels.

    Returns a convolution of a 2D array with the 1D horizontal and vertical
    kernels. This is equivalent to convolving with a 2D kernel equivallent to
    hkern * wkern, but this implementation is faster, solely because it performs
    an horizontal convolution,then a vertical one, effectively reducing a
    quadratic time complexity w.r.t kernel size to a linear one.

    Useful for many separable kernels like Gaussian, Sobel filter, etc.

    Note: np.sum(np.multiply()) is used instead of np.dot as numba doesn't
    support dot product between two 1D arrays.

    Args:
        arr (np.ndarray): 2D array which is to be convolved.
        hkern (np.ndarray): 1D kernel to be convolved vertically (row-wise).
        wkern (np.ndarray): 1D kernel to be convolved horizontally(column-wise).

    Returns:
        Convolved result of the array and the kernels.

    """
    h, w = arr.shape
    hkern = np.flip(hkern)
    wkern = np.flip(wkern)
    s: int = hkern.shape[0]
    tmp: np.ndarray = np.zeros(arr.shape, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            tmp[y][x] = np.sum(
                np.multiply(
                    wkern[max(0, s // 2 - x) : min(s, w - x + s // 2)],
                    arr[y, max(0, x - s // 2) : min(x + s // 2 + 1, w)],
                )
            )
    res: np.ndarray = np.zeros(tmp.shape, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            res[y][x] = np.sum(
                np.multiply(
                    hkern[max(0, s // 2 - y) : min(s, h - y + s // 2)],
                    tmp[max(0, y - s // 2) : min(y + s // 2 + 1, h), x],
                )
            )
    return res

@njit
def convolve3d_separable(
    arr: np.ndarray, hkern: np.ndarray, wkern: np.ndarray
) -> np.ndarray:
    """Convolves a 3D array (color image) with a two 1D kernel.

    Convolves each color channel of a 3D array with the horizontal and vertical
    kernels. Only works with separable 2D kernels.

    Args:
        arr (np.ndarray): 3D array to convolve.
        hkern (np.ndarray): Vertical convolution kernel.
        wkern (np.ndarray): Horizontal convolution kernel.

    Returns:
        3D convolution of the array and the kernels.

    """
    _, _, c = arr.shape
    res: np.ndarray = np.zeros(arr.shape, dtype=np.float64)
    for ch in range(c):
        res[:, :, ch] = convolve2d_separable(arr[:, :, ch], hkern, wkern)
    return res

@njit
def gaussian_filtering(im: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian Filters a noisy color image, represented as a 3D array.

    Performs Gaussian smoothing of a noisy image with a distribution with given
    standard deviation σ.

    It first generates an 1D Gaussian Kernel with given σ, and then applies a
    convolution of it with the 3D array.

    Args:
        im (np.ndarray): 3D array representation of an image to be smoothed.
        sigma (float): The standard deviation of the smoothing kernel.

    Returns:
        Gaussian smoothed 3D array, which can be converted to a color image.

    """
    kern1d: np.ndarray = gaussian_kernel(sigma)
    res: np.ndarray = convolve3d_separable(im, kern1d, kern1d)
    res = np.clip(np.rint(res), 0, 255).astype(np.uint8)
    return res

@njit
def median_filtering(im: np.ndarray, wsz: int = 3) -> np.ndarray:
    """Performs median filtering on a noisy image.

    Replaces the current value of an array at (i, j) with the median of values
    inside a window of size wsz*wsz, centred at (i, j). For mild gaussian noise,
    this filter performs quite good in filtering out noise.

    Args:
        im (np.ndarray): Noisy array(image) to be filtered.
        wsz (int): Side length of the window to be centred at index to find the
          median.

    Returns:
        A filtered array, which is a representation of a color image.

    """
    h, w, c = im.shape
    res: np.ndarray = np.zeros(im.shape, dtype=np.uint8)
    pad: int = wsz // 2
    for ch in range(c):
        for y in range(h):
            for x in range(w):
                res[y][x][ch] = np.median(
                    im[max(y - pad, 0) : y + pad + 1, max(x - pad, 0) : x + pad + 1, ch]
                )
    return res

@njit
def row_lgt(im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Performs Le Gall transform and downsampling in every row.

    For each row of the image array, it first uses the Le Gall transform, and
    then breaks into low and high bands. Scaling by 1/sqrt(2) and sqrt(2) for
    low and high respectively is required to be precise, but it isn't
    implemented here. Instead, that is directly done after both the transforms.

    Args:
        im (np.ndarray): The image array for Le Gall transformation.

    Returns:
        A tuple containg low and high bands respectively. If the initial size of
        the array was h x w, now two h x (w/2) rrays are returned.

    """
    im = im.astype(np.float64)
    w: int = im.shape[1]
    alpha: float = -0.5
    beta: float = 0.25
    for i in range(1, w - 2, 2):
        im[:, i, :] += alpha * (im[:, i - 1, :] + im[:, i + 1, :])
    im[:, w - 1, :] += 2 * alpha * im[:, w - 2, :]
    for i in range(2, w, 2):
        im[:, i, :] += beta * (im[:, i - 1, :] + im[:, i + 1, :])
    im[:, 0, :] += 2 * beta * im[:, 1, :]
    return im[:, ::2, :], im[:, 1::2, :]

@njit
def col_lgt(im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Performs Le Gall transform and downsampling in every column.

    For each column of the image array, it first uses the Le Gall transform, and
    then breaks into low and high bands. Scaling by 1/sqrt(2) and sqrt(2) for
    low and high respectively is required to be precise, but it isn't
    implemented here. Instead, that is directly done after applying both the
    row and column transforms.

    Args:
        im (np.ndarray): The image array for Le Gall transformation.

    Returns:
        A tuple containg low and high bands respectively. If the initial size of
        the array was h x w, now two (h/2) x w arrays are returned.

    """
    im = im.astype(np.float64)
    h: int = im.shape[0]
    alpha: float = -0.5
    beta: float = 0.25
    for i in range(1, h - 2, 2):
        im[i, :, :] += alpha * (im[i - 1, :, :] + im[i + 1, :, :])
    im[h - 1, :, :] += 2 * alpha * im[h - 2, :, :]
    for i in range(2, h, 2):
        im[i, :, :] += beta * (im[i - 1, :, :] + im[i + 1, :, :])
    im[0, :, :] += 2 * beta * im[1, :, :]
    return im[::2, :, :], im[1::2, :, :]

def legall(
    im: np.ndarray, threshold: dict[str, float] = {"LH": 0, "HL": 0, "HH": 0}
) -> dict[str, np.ndarray]:
    """Applies Le Gall 5/3 transform on a color image-array.

    Performs both row and column Le Gall 5/3 transform, and thus separates the
    image array into four bands- LL, LH, HL, HH. The LH, HL ,HH bands can be
    called the detail coefficients, but technically only HH is the details band
    the one that is subject to distortion by noise. Thus, we can apply
    thresholds, so that we can eliminate noises. By default, this was made to
    be completely lossless. Also, the scaling is done here. The LL band has 2
    low bands, thus divided by 2 {(1/√2)^2}, and HH band should be multiplied by
    2 {(√2)^2}. For the other bands. the coefficients are cancelled out.

    Args:
        im (np.ndarray): An array to which row and column Le Gall transform is
          applied.
        th (dict[str, float]): Hard thresholds which are to be applied for
          reducing noise. Note: Only HH band is significant. LL band is the
          least significant for reducing noise, so no threshold for LL band is
          passed.

    Returns:
        A dictionary containg all four bands - LL, LH, HL, HH. The thresholds
        are applied before, so the image has reduced noise in the inverse
        transform.

    """
    lo, hi = row_lgt(im)
    LL, LH = col_lgt(lo)
    HL, HH = col_lgt(hi)
    LL /= 2
    HH *= 2
    LH[np.abs(LH) < threshold["LH"]] = 0
    HL[np.abs(HL) < threshold["HL"]] = 0
    HH[np.abs(HH) < threshold["HH"]] = 0
    return {"LL": LL, "LH": LH, "HL": HL, "HH": HH}

@njit
def col_ilgt(lb: np.ndarray, hb: np.ndarray, rf: float = 0) -> np.ndarray:
    """Performs Inverse Le Gall transform columnwise.

    For each column of the image array, it first applies the Inverse Le Gall
    transform, and then breaks into low and high bands. Scaling by sqrt(2) and
    1/sqrt(2) for low and high respectively is required to be precise, but it
    isn't implemented here. Instead, that is directly done before applying both
    the row and column inverse transforms, in the ilegall function.

    Args:
        lb (np.ndarray): A low band transform.
        hb (np.ndarray): Corresponding high band transform.
        rf (float): If the bands had previously been stored as integers, this is
          a rounding factor for better rounding.

    Returns:
        The array combining the low and the high bands. If the two bands had
        h x w size, now a (2h) x w sized array is returned.

    """
    h, w, c = lb.shape
    arr = np.zeros((2 * h, w, c), dtype=np.float64)
    h *= 2
    arr[::2, :, :] = lb
    arr[1::2, :, :] = hb
    beta = -0.25
    alpha = 0.5
    for i in range(2, h, 2):
        arr[i, :, :] += beta * (arr[i - 1, :, :] + arr[i + 1, :, :]) + rf
    arr[0, :, :] += 2 * beta * arr[1, :, :] + rf
    for i in range(1, h - 2, 2):
        arr[i, :, :] += alpha * (arr[i - 1, :, :] + arr[i + 1, :, :])
    arr[h - 1, :, :] += 2 * alpha * arr[h - 2, :, :]
    return arr

@njit
def row_ilgt(lb: np.ndarray, hb: np.ndarray, rf: float = 0) -> np.ndarray:
    """Performs Inverse Le Gall transform row-wise.

    For each row of the image array, it first applies the Inverse Le Gall
    transform, and then breaks into low and high bands. Scaling by sqrt(2) and
    1/sqrt(2) for low and high respectively is required to be precise, but it
    isn't implemented here. Instead, that is directly done before applying both
    the row and column inverse transforms, in the ilegall function.

    Args:
        lb (np.ndarray): A low band transform.
        hb (np.ndarray): Corresponding high band transform.
        rf (float): If the bands had previously been stored as integers, this is
          a rounding factor for better rounding.

    Returns:
        The array combining the low and the high bands. If the two bands had
        h x w size, now a h x (2w) sized array is returned.

    """
    h, w, c = lb.shape
    arr = np.zeros((h, 2 * w, c), dtype=np.float64)
    w *= 2
    arr[:, ::2, :] = lb
    arr[:, 1::2, :] = hb
    beta = -0.25
    alpha = 0.5
    for i in range(2, w, 2):
        arr[:, i, :] += beta * (arr[:, i - 1, :] + arr[:, i + 1, :]) + rf
    arr[:, 0, :] += 2 * beta * arr[:, 1, :] + rf
    for i in range(1, w - 2, 2):
        arr[:, i, :] += alpha * (arr[:, i - 1, :] + arr[:, i + 1, :])
    arr[:, w - 1, :] += 2 * alpha * arr[:, w - 2, :]
    return arr

def ilegall(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Composes a image from it sub-bands.

    Performs the complete Inverse Le Gall transform. The scaling factors are
    reverted. Also checks if any rounding factor is required or not.

    Args:
        bands (dict[str, np.ndarray]): A dictionary containing the LL, LH, HL
          and HH bands. The image is composed from these.

    Returns:
        An image composed from the bands.

    """
    rf = 0.5 if np.issubdtype(bands["LL"].dtype, np.integer) else 0
    lo = col_ilgt(
        bands["LL"].astype(np.float64) * 2, bands["LH"].astype(np.float64), rf
    )
    hi = col_ilgt(
        bands["HL"].astype(np.float64), bands["HH"].astype(np.float64) / 2, rf
    )
    img = row_ilgt(lo, hi, rf)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)

def psnr(im: np.ndarray, noisy: np.ndarray) -> np.float64:
    """Computes PSNR of the original and noisy image (array).

    Peak signal-to-noise ratio or PSNR is the ratio between the maximum possible
    power of a signal and the power of corrupting noise that affects the
    fidelity of its representation. It is expressed as an logarithmic quantity
    using the decibel scale. In plain terms, more the PSNR of a noisy/filtered
    image with the original image, better does the noisy/filtered image
    represents the original image. PSNR of a very noisy image and original image
    would be quite low, while the PSNR of the original image with the original
    image would be infinity.

    Args:
        im (np.ndarray): Array representaion of the original image.
        noisy (np.ndarray): Array representation of noisy image.

    Returns:
        PSNR of the original and noisy image.

    """
    MSE: np.float64 = np.mean(np.square(np.subtract(im, noisy, dtype=np.float64)))
    return 20 * np.log10(255) - 10 * np.log10(MSE)

@njit
def histogram_equalization(im: np.ndarray) -> np.ndarray:
    """Histogram equalizes a grayscale image.

    Performs histogram equalization on a grayscale image(array). This method is
    used to adjust contrast in a grayscale image.

    Args:
        im (np.ndarray): Array representation of a grayscale image.

    Returns:
        Histogram equalized image.
    """
    imf: np.ndarray = im.flatten()
    hist: np.ndarray = np.bincount(imf, minlength=256).astype(np.float64)
    hist /= imf.shape[0]
    cdf: np.ndarray = np.copy(hist)
    for i in range(1, 256):
        cdf[i] += cdf[i - 1]
    cdf = np.ceil(cdf * 256) - 1
    cdf = np.clip(cdf, 0, 255).astype(np.uint8)
    res: np.ndarray = np.zeros(imf.shape, dtype=np.uint8)
    for i in range(256):
        res[imf == i] = cdf[i]
    return res.reshape(im.shape)

@njit
def otsu_util(tot: np.int64, mean: np.int64, hist: np.ndarray) -> int:
    """Returns the threshold of the otsu thresholding method.

    Performs an automatic image thresholding. It divides the image into two
    classes, the foreground and background. For more detailed description of
    the algorithm, including multi-class classification, read "A threshold
    selection method from gray-level histograms" by Nobuyuki Otsu.

    Args:
        tot (np.int64): Total number of pixels or array element.
        mean (np.int64): Total class mean.
        hist (np.ndarray): Histogram of all pixel intensities of an image.

    Returns:
        The threshold intensity to perform Otsu binarization.

    """
    maxvar: np.float64 = np.float64(-1)
    threshold: int = 0
    omega0: np.int64 = hist[0]
    muo0: np.int64 = np.int64(0)
    for i in range(1, 255):
        omega1: np.int64 = tot - omega0
        muo1: np.int64 = mean - muo0
        if omega0 > 0 and omega1 > 0:
            var: np.float64 = omega0 * omega1 * np.square(muo0 / omega0 - muo1 / omega1)
            if var >= maxvar:
                maxvar = var
                threshold = i
        omega0 += hist[i]
        muo0 += i * hist[i]
    return threshold

def otsu(im: np.ndarray) -> np.ndarray:
    """Computes an Otsu binarized grayscale image.

    Computes the total class mean, total number of pixels, and then perform
    binarization with the threshold value from the otsu_util function.

    The Otsu's method was divided into two functions because of problems with
    numba like no available support of np.dot and np.where for 2D arrays.

    Args:
        im (np.ndarray): The image on which Otsu's method is performed.

    Returns:
        Otsu binarized image.

    """
    hist: np.ndarray = np.bincount(im.flatten(), minlength=256)
    tot: np.int64 = np.int64(im.shape[0] * im.shape[1])
    mean: np.int64 = np.dot(np.arange(256), hist)
    threshold: int = otsu_util(tot, mean, hist)
    res: np.ndarray = im.copy()
    res[im > threshold] = 255
    res[im < threshold] = 0
    return res

@njit
def zero_crossing(arr: np.ndarray, zth: float = 0) -> np.ndarray:
    """Computes zero crossings of a LoG convolved image.

    The function finds elements in an array, who have two differently signed
    elements on it either side. In other words, we find locations where the sign
    changes and assign those places a bright intensity pixel, and others 0.
    This way we find the edges.

    Args:
        arr (np.ndarray): The LoG kernel applied array.
        zth (float): The threshold which can be used instead of 0. We may ignore
         places with small sign changes, hence this threshold.

    Returns:
        Zero crossing computed image, which makes us easier to find edges.

    """
    h, w = arr.shape
    res: np.ndarray = np.zeros(arr.shape, dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if (
                arr[y - 1][x - 1] * arr[y + 1][x + 1] < zth
                or arr[y - 1][x + 1] * arr[y + 1][x - 1] < zth
                or arr[y + 1][x] * arr[y - 1][x] < zth
                or arr[y][x + 1] * arr[y][x - 1] < zth
            ):
                res[y][x] = 255
    return res

@njit
def harris_util(
    im: np.ndarray, k: float = 0.05, sigma_1: float = 1, sigma_2: float = 3
) -> np.ndarray:
    """Computes the array on which thresholding can be done to find corners.

    Performs gaussian smoothing on image,then applies sobel filters to find
    gradients. Applies gaussian smoothing on autocorrelation coefficient
    matrices too, otherwise the corner detection doesn't work as expected. R > 0
    means corners, and <0 means edges, and approximately 0 means flat. But it
    does not work exactly that way, so threshold for corner detection is
    provided. If either of the smoothing is not done, it underperforms, and if
    either is heavily smoothed, then too it underperforms. It is hard to find
    the optimal values for smoothing. Later, in places where there is a local
    maximum, a 3x3 window is marked white centering the maximum so that the dots
    are easily visible.

    Args:
        im (np.ndarray): Array representation of image for corner ddtection.
        k (float): Harris coefficient. k ⋴ [0.04, 0.06] generally.
        sigma_1 (float) : Standar deviation σ for Gaussian image smoothing.
        sigma_2 (float) : Standar deviation σ for Gaussian Autocorrelation
          coefficient smoothing.

    Returns:
        An array with a window of 3x3 of all local maximums.

    """
    gauss_1: np.ndarray = gaussian_kernel(sigma_1)
    im = convolve2d_separable(im, gauss_1, gauss_1)
    sobel_p1: np.ndarray = np.array([1, 2, 1])
    sobel_p2: np.ndarray = np.array([-1, 0, 1])
    Ix: np.ndarray = convolve2d_separable(im, sobel_p1, sobel_p2) / 8
    Iy: np.ndarray = convolve2d_separable(im, -sobel_p2, sobel_p1) / 8
    gauss_2: np.ndarray = gaussian_kernel(sigma_2)
    Ixx: np.ndarray = convolve2d_separable(np.square(Ix), gauss_2, gauss_2)
    Ixy: np.ndarray = convolve2d_separable(np.multiply(Ix, Iy), gauss_2, gauss_2)
    Iyy: np.ndarray = convolve2d_separable(np.square(Iy), gauss_2, gauss_2)
    R: np.ndarray = np.multiply(Ixx, Iyy) - np.square(Ixy) - k * np.square(Ixx + Iyy)
    h, w = im.shape
    res = np.zeros(im.shape, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            if R[y][x] > 0:
                xl = max(0, x - sigma_2)
                xr = min(x + sigma_2 + 1, w)
                yl = max(0, y - sigma_2)
                yr = min(y + sigma_2 + 1, h)
                if R[y][x] == np.amax(R[yl:yr, xl:xr]):
                    res[yl:yr, xl:xr] = R[y][x]
    return res

def harris(
    im: np.ndarray,
    k: float = 0.05,
    sigma_1: float = 1,
    sigma_2: float = 3,
    threshold: float = 1,
) -> np.ndarray:
    """Thresholds and return all corners in an image.

    Performs thresholding, while all other computation is done in the
    harris_util function. This function exists because of numba not supporting
    the 2D thresholding part.

    Args:
        im (np.ndarray): Array representation of image for corner ddtection.
        k (float): Harris coefficient. k ⋴ [0.04, 0.06] generally.
        sigma_1 (float) : Standar deviation σ for Gaussian image smoothing.
        sigma_2 (float) : Standar deviation σ for Gaussian Autocorrelation
          coefficient smoothing.
        threshold: The threshold above which all R values are considered as
          maximum. Others are ignored.

    Returns:
        All corners in a grayscale image.
    """
    res = harris_util(im, k, sigma_1, sigma_2)
    res[res >= threshold] = 255
    res[res < threshold] = 0
    return res.astype(np.uint8)

def save_npy(arr: np.ndarray, file: Path) -> None:
    """Save npy files, as they are extremely fast to retrieve.

    It was required since many testing were done on many other images as well.
    Saves split seconds of time generally,but maybe useful if there are too
    many large images to test on.

    Args:
        arr (np.ndarray): Array to be saved as numpy file.
        file (Path): Name and path of the file to be saved.
    """
    with open(file, "wb") as f:
        np.save(f, arr)

def save_img(arr: np.ndarray, file: Path) -> None:
    """Saves image in a file.
    arr (np.ndarray): Array to be saved as a Image.
    file (Path): Filename and path where the array is to be saved.
    """
    img = Image.fromarray(arr)
    img.save(file, "jpeg")
