import copy
import numpy as np
import cv2
import ocrodeg


def random_blotches(image, fgblobs, bgblobs, fgscale=5, bgscale=5):
    fg = (ocrodeg.random_blobs(image.shape[:2], fgblobs, fgscale) * 255) [:, : , np.newaxis]
    bg = 255 - (ocrodeg.random_blobs(image.shape[:2], bgblobs, bgscale) * 255) [:, : , np.newaxis]
    fg = fg.astype(np.uint8)
    bg = bg.astype(np.uint8)
    return np.minimum(np.maximum(image, fg), bg)


def apply_speckle(image, degree, density_basic = 1e-4, save_path=None):
    if degree == 0:
        return image
    degree = 2 * degree - 1
    density = density_basic * degree
    image_with_blotches = random_blotches(image, density, density)
    if save_path is not None:
        cv2.imwrite(save_path, image_with_blotches)
        return image_with_blotches
    return image_with_blotches



def apply_texture(image, degree, save_path=None):
    if degree == 0:
        return image
    degree = 2 * degree - 1
    num_fibers = int(degree * 300)
    image_shape = image.shape[:2]
    scale_image = copy.deepcopy(image).astype(np.float64)
    scale_image -= np.min(scale_image)
    if np.max(scale_image) != 0:
        scale_image /= np.max(scale_image)
    else:
        save_path = "log/abnormaly/" + str(np.random.randint(0, 10000))
        save_ori = save_path + "_ori.jpg"
        save_path = save_path + ".jpg"
        cv2.imwrite(save_ori, image)
    selector = ocrodeg.autoinvert(scale_image)
    paper = ocrodeg.make_fibrous_image(image_shape, num_fibers, 500, 0.01, limits=(0.0, 0.25), blur=0.5)
    ink = ocrodeg.make_multiscale_noise(image_shape, [1.0, 5.0, 10.0, 50.0], limits=(0.0, 0.5))
    paper = np.stack([paper] * 3, axis=-1)
    ink = np.stack([ink] * 3, axis=-1)
    printed = (selector * ink + (1 - selector) * (1-paper)) * 255
    printed = printed.astype(np.uint8)
    if save_path is not None:
        cv2.imwrite(save_path, printed)
        return printed
    return printed
