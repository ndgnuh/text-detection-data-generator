import random

import cv2
import numpy as np


def generate_random_color():
    # Generate random RGB color values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def check_suitability(background_image, text_color, max_diff=150):
    """
    Check the suitability of the text color for the given background image.

    Args:
        background_image: The background image as a NumPy array.
        text_color: The RGB color value of the text.
        max_diff (int): The maximum lightness difference between the text color and the background image.
                        Default is 150.

    Returns:
        bool: True if the text color is suitable, False otherwise.

    """

    # Convert the background image to HSL color space
    hsl_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2HLS)

    # Calculate the average lightness value of the background image
    average_lightness = np.mean(hsl_image[:, :, 1])

    # Convert the text color to HSL color space
    hsl_text_color = cv2.cvtColor(np.uint8([[text_color]]), cv2.COLOR_BGR2HLS)[0][0]

    # Check the suitability of the text color based on lightness value
    if abs(hsl_text_color[1] - average_lightness) > max_diff:
        return False
    else:
        return True


def gen_text_color(background_image: np.ndarray, max_diff=150):
    """
    Generate a suitable text color for the given background image.

    Args:
        background_image (np.ndarray): The background image as a NumPy array.
        max_diff (int): The maximum lightness difference between the text color and the background image.
                        Default is 150.

    Returns:
        Tuple[int, int, int]: The RGB color values of the text color.

    """
    count_loop = 0
    while True:
        if count_loop > 1500:
            # raise ValueError("Can't find suitable text color")
            return None

        text_color = generate_random_color()
        if check_suitability(background_image, text_color, max_diff=max_diff):
            return text_color
        else:
            count_loop += 1


def gen_text_color_v2(bg_img: np.ndarray, min_text_bg_rate=4.5, max_loop=1000):
    """
    Generate a suitable text color for the given background image using an alternative method.

    Args:
        bg_img (np.ndarray): The background image as a NumPy array.
        min_text_bg_rate (float): The minimum text to background lightness ratio. Default is 4.5.
        max_loop (int): The maximum number of loop iterations. Default is 1000.

    Returns:
        Tuple[int, int, int]: The RGB color values of the text color.

    """
    mean_color = cv2.mean(bg_img)
    r, g, b = mean_color[2] / 255, mean_color[1] / 255, mean_color[0] / 255
    # Sinh màu ngẫu nhiên
    text1_r = random.uniform(0, 1)
    text1_g = random.uniform(0, 1)
    text1_b = random.uniform(0, 1)

    # Tính độ khác nhau so với bg_color
    contrast1 = (0.299 * r + 0.587 * g + 0.114 * b + 0.05) / (
        0.299 * text1_r + 0.587 * text1_g + 0.114 * text1_b + 0.05
    )

    # Nếu chưa đủ khác, thực hiện sinh lại
    count = 0
    while contrast1 < min_text_bg_rate:
        count += 1
        if count > max_loop:
            # raise ValueError("Can't find suitable text color")
            return None
        # print("count: ",count, ". contrast1: ", contrast1)
        text1_r = random.uniform(0, 1)
        text1_g = random.uniform(0, 1)
        text1_b = random.uniform(0, 1)
        contrast1 = (0.299 * r + 0.587 * g + 0.114 * b + 0.05) / (
            0.299 * text1_r + 0.587 * text1_g + 0.114 * text1_b + 0.05
        )

    new_color = (int(text1_r * 255), int(text1_g * 255), int(text1_b * 255))
    return new_color
