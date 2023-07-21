from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


def fill_text_to_image(
    image: Image,
    text: str,
    font: ImageFont,
    textsize: int,
    box: list,
    text_color: Tuple[int, int, int] = (0, 0, 0),
):
    """
    Fill the given text onto the image within the specified bounding box.

    Args:
        image (Image): The input image.
        text (str): The text to fill.
        font (ImageFont): The font to use for the text.
        textsize (int): The font size.
        box (list): The bounding box coordinates as a list [x1, y1, x2, y2].
        text_color (Tuple[int, int, int]): The RGB color value of the text. Default is (0, 0, 0) (black).

    Returns:
        Image: The modified image with the filled text.

    """
    font_copy = font.font_variant(size=textsize)
    # text_mask = Image.new("RGB", (crop_width, img_height), color = bg_color)
    draw = ImageDraw.Draw(image)

    # text_y=abs(img_height-crop_height)
    draw.text((int(box[0]), int(box[1])), text, font=font_copy, fill=text_color)

    return image


def draw_rectangle(image, x1, y1, x2, y2, line_width=2):
    """
    Draw a rectangle on the image with the specified coordinates.

    Args:
        image (Image): The input image.
        x1 (int): The x-coordinate of the top-left corner of the rectangle.
        y1 (int): The y-coordinate of the top-left corner of the rectangle.
        x2 (int): The x-coordinate of the bottom-right corner of the rectangle.
        y2 (int): The y-coordinate of the bottom-right corner of the rectangle.
        line_width (int): The width of the rectangle's outline. Default is 2.

    Returns:
        Image: The modified image with the drawn rectangle.

    """

    # Create an ImageDraw object
    draw = ImageDraw.Draw(image)

    # Draw the rectangle
    rectangle = [(x1, y1), (x2, y2)]
    draw.rectangle(rectangle, outline="red", width=line_width)

    return image
