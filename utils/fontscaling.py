import cv2
from pathlib import Path
import numpy as np


def write_heading(img, text, x, y, font_size = 0.6, border = True):
    text_thickness = int(round(font_size * 2.25))
    if border:
        border_thickness = int(round(2 * text_thickness))
        cv2.putText(img=img, text=text, org=(x, y), fontFace=2, fontScale=font_size, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
    cv2.putText(img=img, text=text, org=(x, y), fontFace=2, fontScale=font_size, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)

def write_text(img, text, x, y, font_size = 0.45, border = True):
    if border:
        text_thickness = int(round(font_size * 2.25))
        border_thickness = int(round(2 * text_thickness))
        cv2.putText(img=img, text=text, org=(x, y), fontFace=0, fontScale=font_size, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
    cv2.putText(img=img, text=text, org=(x, y), fontFace=0, fontScale=font_size, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)

def test_image(out_path, h, w):
    heading_text = "yolor-edge / Ewan Thompson / B.Eng(Hons) Thesis / 2021"

    img = np.zeros((h,w,3),dtype=np.uint8)


    # Heading
    write_heading(img, heading_text, 10,10)
    write_text(img, "Instantaneous:", 20, 20, 0.325)
    write_text(img, "FPS:   3.25", 20, 30, 0.325)
    write_text(img, "NMS:   3.21ms", 20, 40, 0.325)
    write_text(img, "Inf:   4.21ms", 20, 50, 0.325)


    # cv2.putText(img=img, text=text_line, org=(x, y), fontFace=font_face, fontScale=font_scale, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
    # cv2.putText(img=img, text=text_line, org=(x, y), fontFace=font_face, fontScale=font_scale, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)

    
    img_path =Path(out_path).joinpath(f"test-{w}x{h}.jpg")
    print(f"Writing to {img_path}")
    cv2.imwrite(str(img_path), img)

def create_img_set(out_path, h, w):

    text = "yolor-edge / Ewan Thompson / B.Eng(Hons) Thesis / 2021"
    img = np.zeros((h,w,3),dtype=np.uint8)

    line = 1
    x = 10
    y = 0
    scale = 0.

    range_start = 48
    range_end = 66

    pixels = w * h
    if pixels > 2e6:
        range_start = 0.65
        range_end = 2


    for i in range(range_start,range_end,10):
        scale = i / 100.
        y =scale * line * 50
        if y >= h:
            line = 0
            x += scale * 75
            if x > w:
                break
            y = 0


        font_scale = scale
        text_thickness = int(round(font_scale * 2.25))
        border_thickness = int(round(2 * text_thickness))

        x = int(x)
        y = int(y)

        text_line = f"{text} {scale:.3f}"

        cv2.putText(img=img, text=text_line, org=(x, y), fontFace=2, fontScale=font_scale, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img=img, text=text_line, org=(x, y), fontFace=2, fontScale=font_scale, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)
        line += 1

    
    img_path =Path(out_path).joinpath(f"test-{w}x{h}.jpg")
    print(f"Writing to {img_path}")
    cv2.imwrite(str(img_path), img)


if __name__ == "__main__":
    outpath = Path("/resources/tests/fontscaling")

    if not outpath.exists():
        outpath.mkdir(parents=True,exist_ok=True)

    image_sizes = [
        [640,480],
        [1024,768],
        [1080,1920],
        [1280,720],
        [1280,1024],
        [1920,1080],
        [2048,1080],
        [2160,3840],
        [2160,4096],
        [2560,1440],
        [3840,2160],
        [4096,2160],
        [7680,4320],
    ]
    ff = 6
    more_image_sizes = []
    for img_sz in image_sizes:
        w,h = img_sz
        create_img_set(str(outpath),h=h,w=w)
        create_img_set(str(outpath),h=w,w=h)

