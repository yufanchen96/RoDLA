import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
import os
import json


def create_background_file(backimage_path):
    file_list = [f for f in os.listdir(backimage_path) if os.path.isfile(os.path.join(backimage_path, f))]
    save_path = backimage_path + "background.json"
    with open(save_path, "w") as f:
        json.dump(file_list, f)


def add_watermark(image, degree, save_path=None, watermark_text=None, rotation=True):
    if degree == 0:
        return image
    if watermark_text is None:
        watermark_text = "Watermark_just_for_test_in_order_to_see_if_this_works"
    degree = 2 * degree - 1
    scale_factor = degree + 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    h, w = image.shape[:2]
    watermark_image = Image.new("RGBA", (int(w/scale_factor), int(h/scale_factor)), (0, 0, 0, 0))
    position = (np.random.randint(0, int(w/scale_factor)), np.random.randint(0, int(h/scale_factor)))
    font_color = (255, 0, 0)
    watermark_alpha = int(degree * (255/5))
    draw = ImageDraw.Draw(watermark_image)
    draw.text(position, watermark_text, fill=(*font_color, watermark_alpha))
    watermark_image = watermark_image.resize((w, h), Image.LANCZOS)
    if rotation:
        rotation_angle = np.random.randint(0, 360)
        watermark_image = watermark_image.rotate(rotation_angle, expand=False)
    watermarked_image = Image.alpha_composite(pil_image.convert("RGBA"), watermark_image)
    image_with_watermark = cv2.cvtColor(np.array(watermarked_image), cv2.COLOR_RGBA2BGR)
    if save_path is not None:
        cv2.imwrite(save_path, image_with_watermark)
        return image_with_watermark
    return image_with_watermark




def add_image_background(image, degree, background_folder=None, save_path=None):
    if degree == 0:
        return image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image).convert("RGBA")
    h, w = image.shape[:2]
    background_numbers = 2 * degree - 1
    alpha_layer_original = Image.new("L", pil_image.size, 255)
    pil_image.putalpha(alpha_layer_original)
    background_image = pil_image.copy()
    if background_folder is None:
        background_path = "/".join(os.path.abspath(__file__).split("/")[:-1]) + "/background_image/"
    else:
        background_path = background_folder
    if os.path.exists(background_path + "background.json") is False:
        create_background_file(background_path)
    file_list = json.load(open(background_path + "background.json", "r"))
    for _ in range(background_numbers):
        file_name = random.choice(file_list)
        b_path = os.path.join(background_path, file_name)
        back_image = cv2.imread(b_path, cv2.IMREAD_UNCHANGED)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)
        back_image = Image.fromarray(back_image)
        back_image = back_image.resize((int(w/5), int(h/5)), Image.LANCZOS)
        safe_area_w, safe_area_h = w - back_img_w, h - back_img_h
        offset_w, offset_h = int(safe_area_w * 0.4), int(safe_area_h * 0.4)
        position = (
            np.random.randint(offset_w, safe_area_w - offset_w),
            np.random.randint(offset_h, safe_area_h - offset_h)
        )
        background_image.paste(back_image, position)
    alpha_layer = Image.new("L", background_image.size, 100)
    background_image.putalpha(alpha_layer)
    image_with_background = Image.alpha_composite(pil_image, background_image)
    image_with_background = cv2.cvtColor(np.array(image_with_background), cv2.COLOR_RGBA2BGR)
    if save_path is not None:
        cv2.imwrite(save_path, image_with_background)
        return image_with_background
    return image_with_background



