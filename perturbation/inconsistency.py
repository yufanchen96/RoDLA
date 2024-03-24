
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter


def apply_ink_holdout(image, degree, iterations=1, save_path=None):
    if degree == 0:
        return image
    degree = 2 * degree - 1
    kernel_size = 2 * degree + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    large_image_shape = (image.shape[1] * 10, image.shape[0] * 10)
    image_large = cv2.resize(image, large_image_shape, interpolation=cv2.INTER_LINEAR)
    image_with_erosion_l = cv2.erode(image_large, kernel, iterations=iterations)
    image_with_erosion = cv2.resize(image_with_erosion_l, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    if save_path is not None:
        cv2.imwrite(save_path, image)
        return image_with_erosion
    return image_with_erosion


def apply_ink_bleeding(image, degree, iterations=1, save_path=None):
    if degree == 0:
        return image
    degree = 2 * degree - 1
    kernel_size = 2 * degree + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    large_image_shape = (image.shape[1] * 10, image.shape[0] * 10)
    image_large = cv2.resize(image, large_image_shape, interpolation=cv2.INTER_LINEAR)
    image_with_dilation_l = cv2.dilate(image_large, kernel, iterations=iterations, anchor=(0, 0) )
    image_with_dilation = cv2.resize(image_with_dilation_l, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    if save_path is not None:
        cv2.imwrite(save_path, image)
        return image_with_dilation
    return image_with_dilation


def apply_illumination(image, degree, save_path=None):
    if degree == 0:
        return image
    degree = 2 * degree - 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image).convert("RGB")
    mask = Image.new('L', pil_image.size, color=255)
    draw = ImageDraw.Draw(mask)
    points = [(np.random.randint(mask.width), np.random.randint(mask.height)) for _ in range(5)]
    draw.polygon(points, fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=50))
    shadow_light_switch = np.random.randint(2)
    if shadow_light_switch == 0:
        darker_image = ImageEnhance.Brightness(pil_image).enhance(255/10*(degree+1))
    else:
        darker_image = ImageEnhance.Brightness(pil_image).enhance(1/(degree+1))
    shadowed_image = Image.composite(pil_image, darker_image, mask)
    shadowed_image = cv2.cvtColor(np.array(shadowed_image), cv2.COLOR_RGB2BGR)
    if save_path is not None:
        cv2.imwrite(save_path, shadowed_image)
        return shadowed_image
    return shadowed_image
