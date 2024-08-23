import os
import random
import numpy as np
import cv2

from tqdm import tqdm
from imgaug import augmenters as iaa
from PIL import Image, ImageFont, ImageDraw
from glob import glob

CANVAS_DARK_RANGE = 75
CANVAS_RANGE_LIGHT = 215
FONTS_DIR = "data/fonts"
ALL_FONT_FPS = glob(FONTS_DIR + "/*.ttf")


def get_random_time_remaining_str():
    minutes_remaining = random.randint(0, 11)
    seconds_remaining = random.randint(0, 59)
    if minutes_remaining == 0:
        deca_seconds = random.randint(0, 9)
        return f"{seconds_remaining:02d}.{deca_seconds}"
    else:
        return f"{minutes_remaining:02d}:{seconds_remaining:02d}"


def get_random_canvas_text_colors():
    coin_flip = random.randint(0, 1)
    text_color = (0, 0, 0)
    canvas_color = (0, 0, 0)
    # light colored canvas
    if coin_flip == 0:
        canvas_color = (
            random.randint(CANVAS_RANGE_LIGHT, 255),
            random.randint(CANVAS_RANGE_LIGHT, 255),
            random.randint(CANVAS_RANGE_LIGHT, 255),
        )
        text_color = (
            random.randint(0, CANVAS_DARK_RANGE),
            random.randint(0, CANVAS_DARK_RANGE),
            random.randint(0, CANVAS_DARK_RANGE),
        )
    else:
        canvas_color = (
            random.randint(0, CANVAS_DARK_RANGE),
            random.randint(0, CANVAS_DARK_RANGE),
            random.randint(0, CANVAS_DARK_RANGE),
        )
        text_color = (
            random.randint(CANVAS_RANGE_LIGHT, 255),
            random.randint(CANVAS_RANGE_LIGHT, 255),
            random.randint(CANVAS_RANGE_LIGHT, 255),
        )
    return canvas_color, text_color


def get_random_canvas_width_height():
    return random.randint(135, 200), random.randint(50, 95)


def generate_rand_roi():
    """
    Return a randomly generated img containg a time-remaining value with the format: `MM:SS.DS`.
    """

    canvas_size = get_random_canvas_width_height()
    canvas_color, text_color = get_random_canvas_text_colors()

    # create a blank image w/ color `canvas color`
    img = Image.new("RGB", canvas_size, canvas_color)

    font_path = random.choice(ALL_FONT_FPS)

    # random fraction of absolute height
    font_size = random.randint(canvas_size[1] - 40, canvas_size[1] - 30)
    font = ImageFont.truetype(font_path, font_size)
    text = get_random_time_remaining_str()

    # draw text in center of image
    draw = ImageDraw.Draw(img)
    _, _, w, h = draw.textbbox((0, 0), text, font=font)
    W, H = canvas_size
    draw.text(((W - w) / 2, (H - h) / 2), text, font=font, fill=text_color)

    # convert to np array
    img = np.array(img)

    PAD = 8
    font_bbx = font.getbbox(text)
    x1, y1, x2, y2 = font_bbx
    bbx_width, bbx_height = (x2 - x1), (y2 - y1)
    center_width, center_height = W / 2, H / 2
    x1, y1, x2, y2 = (
        center_width - (bbx_width / 2),
        center_height - (bbx_height / 2) - PAD,
        center_width + (bbx_width / 2),
        center_height + (bbx_height / 2) + PAD,
    )
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # print(f"Font bbx: {font_bbx}")
    # img = cv2.rectangle(img, (x1, y1), (x2, y2), (155, 155, 235), thickness=1)

    W_PAD = 3
    abs_bbx_width = bbx_width
    bbx_width, _ = int((x2 - x1) / len(text)), (y2 - y1)
    bbxs = []
    for idx, char in enumerate(text):
        width_offset = idx * bbx_width
        x1, y1, x2, y2 = (
            center_width - (abs_bbx_width / 2) + width_offset - W_PAD,
            center_height - (bbx_height / 2) - PAD,
            center_width - (abs_bbx_width / 2) + width_offset + bbx_width + W_PAD,
            center_height + (bbx_height / 2) + PAD,
        )
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbxs.append((char, (x1, y1, x2, y2)))
    #     img = cv2.rectangle(
    #         img,
    #         (x1, y1),
    #         (x2, y2),
    #         (
    #             155 + random.randint(-20, 20),
    #             155 + random.randint(-20, 20),
    #             235 + random.randint(-20, 20),
    #         ),
    #         thickness=1,
    #     )

    # cv2.imwrite("test_roi.png", img)
    return text, img, bbxs


def generate_synth_dataset(out_dir: str, num_samples=100000):
    assert os.path.isdir(out_dir), f"{out_dir} does not exist!"
    for idx in tqdm(
        range(num_samples), total=num_samples, desc="Generating Synth Data"
    ):
        text, img, bbxs = generate_rand_roi()
        out_img_path = os.path.join(out_dir, f"{text}_{idx}.png")
        out_label_path = os.path.join(out_dir, f"{text}_{idx}.txt")
        bbxs = [
            [b[0], str(b[1][0]), str(b[1][1]), str(b[1][2]), str(b[1][3])] for b in bbxs
        ]
        bbxs = [" ".join(b) + "\n" for b in bbxs]
        cv2.imwrite(out_img_path, img)
        with open(out_label_path, "w") as f:
            f.writelines(bbxs)


if __name__ == "__main__":
    generate_synth_dataset("data/synth-roi-dataset")
