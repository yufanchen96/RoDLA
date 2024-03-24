import cv2
import numpy as np


def apply_defocus(image, degree, kernel_size=1, save_path=None):
    if degree == 0:
        return image
    degree =2*degree-1
    kernel_size = int(degree * kernel_size)
    sigma = degree
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    if save_path is not None:
        cv2.imwrite(save_path, blurred_image)
        return blurred_image
    return blurred_image


def apply_vibration(image, degree, base_size=3, save_path=None):
    if degree == 0:
        return image
    degree =2*degree-1
    size = int(degree * base_size)
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size, dtype=np.float32)
    kernel_motion_blur = cv2.warpAffine(
        kernel_motion_blur,
        cv2.getRotationMatrix2D(
            (size / 2 - 0.5, size / 2 - 0.5), np.random.uniform(-45, 45), 1.0
        ),
        (size, size),
    )
    kernel_motion_blur = kernel_motion_blur * (1.0 / np.sum(kernel_motion_blur))
    blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)
    if save_path is not None:
        cv2.imwrite(save_path, blurred_image)
        return blurred_image
    return blurred_image