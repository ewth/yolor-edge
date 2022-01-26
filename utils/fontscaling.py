import cv2
from pathlib import Path
import numpy as np


def write_heading(img, text, x, y, font_size = 0.6, border = True):
    shape = img.shape
    text_thickness = int(round(font_size * 2.25))
    if border:
        border_thickness = int(round(2 * text_thickness))
        cv2.putText(img=img, text=text, org=(x, y), fontFace=2, fontScale=font_size, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
    cv2.putText(img=img, text=text, org=(x, y), fontFace=2, fontScale=font_size, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)
    return img

def write_text(img, text, x, y, font_size = 0.45, border = True):
    shape = img.shape
    if border:
        text_thickness = int(round(font_size * 2.25))
        border_thickness = int(round(2 * text_thickness))
        cv2.putText(img=img, text=text, org=(x, y), fontFace=0, fontScale=font_size, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
    cv2.putText(img=img, text=text, org=(x, y), fontFace=0, fontScale=font_size, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)

    return img

def calc_font_scale(w, h) -> float:

    heading_scale = 0.665

    factor = ((w/1920.0) + (h/1080.0)) / 2.0
    if factor >= 1:
        heading_scale = np.math.floor(factor)

    text_scale = heading_scale / 2.0
    return [heading_scale, text_scale]

def test_image(out_path, h, w):
    heading_text = "yolor-edge / Ewan Thompson / 2021"

    img = np.zeros((h,w,3),dtype=np.uint8)

    heading_size, text_size = calc_font_scale(w, h)

    # Heading
    line_height = int(round(50 * heading_size))
    write_heading(img, heading_text, 10, line_height, heading_size)
    write_heading(img, f"({heading_size:.3f})", 10, line_height * 2, heading_size)
    y = line_height * 3
    line_height = int(round(50 * text_size))
    write_text(img, "Instantaneous:", 10, y + line_height * 4, text_size)
    write_text(img, "FPS:   3.25", 20, y + line_height * 5, text_size)
    write_text(img, "NMS:   3.21ms", 20, y + line_height * 6, text_size)
    write_text(img, "Inf:   4.21ms", 20, y + line_height * 7, text_size)
    write_text(img, f"({text_size:.3f})", 20, y + line_height * 8, text_size)

    # cv2.putText(img=img, text=text_line, org=(x, y), fontFace=font_face, fontScale=font_scale, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
    # cv2.putText(img=img, text=text_line, org=(x, y), fontFace=font_face, fontScale=font_scale, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)
    
    img_path =Path(out_path).joinpath(f"test-{w}x{h}.jpg")
    print(f"Writing to {img_path}")
    cv2.imwrite(str(img_path), img)

def create_img_set(out_path, h, w):

    text_heading = "yolor-edge / Ewan Thompson / B.Eng(Hons) Thesis / 2021"
    # text = f"Source: {w}x{h}px; Testing Font Scaling"
    img = np.zeros((h,w,3),dtype=np.uint8)

    line = 1
    x = 10
    y = 0
    scale = 0.

    range_start = int(250)
    range_end = int(400)
    range_step = 10
    # scale_range = range(range_start,range_end,range_step)
    scale_range = [0.48,0.6,0.6,0.6,0.64,0.66,0.66,1.8,2,1.2,1.5,1.6,0.7,.8,.9,1,1.05,1.1,1.15,0.67,0.68,0.69,.65]
    scale_range = list(set(scale_range))
    scale_range.sort()
    # pixels = w * h
    # if pixels > 2e6:
    #     range_start = 0.65
    #     range_end = 2

    line_height = 50


    for i in scale_range:
        # scale = i / 100.
        scale = i
        y = scale * line * line_height
        # if y >= h:
        #     line = 0
        #     x = x + scale * 75
        #     if x > w:
        #         break
        #     y = line_height


        font_scale = scale
        text_thickness = int(round(font_scale * 2.25))
        border_thickness = int(round(2 * text_thickness))

        x = int(round(x))
        y = int(round(y))

        heading_line = f"{text_heading}"
        text_line = ""
        # text_line = f"{scale:.3f} {text}"
        write_heading(img, heading_line, x, y, font_scale)
        # write_text(img, text_line, line_height, y, font_scale)

        # cv2.putText(img=img, text=heading_line, org=(x, y), fontFace=2, fontScale=font_scale, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
        # cv2.putText(img=img, text=heading_line, org=(x, y), fontFace=2, fontScale=font_scale, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)

        # y = int(round(y + scale * line_height))

        # cv2.putText(img=img, text=text_line, org=(x, y), fontFace=2, fontScale=font_scale, color=(0,0,0), thickness=border_thickness, lineType=cv2.LINE_AA)
        # cv2.putText(img=img, text=text_line, org=(x, y), fontFace=2, fontScale=font_scale, color=(255,255,255), thickness=text_thickness, lineType=cv2.LINE_AA)

        
        line += 1

    
    img_path = Path(out_path).joinpath(f"test-{w}x{h}.jpg")
    print(f"Writing to {img_path}")
    cv2.imwrite(str(img_path), img)


if __name__ == "__main__":
    outpath = Path("/resources/tests/fontscaling")

    if not outpath.exists():
        outpath.mkdir(parents=True,exist_ok=True)

    flip_wh = False

    image_sizes = [
        # Actual sizes encountered so far
        [720,1280],
        [1080,1920],
        [1080,2048],
        [1280,720],
        [1400,1080],
        [1920,1080],
        [2048,1080],
        [2160,3840],
        [2160,4096],
        [2560,1440],
        [3840,2160],
        [4096,2160],
    ]
    ff = 6
    more_image_sizes = []
    for img_sz in image_sizes:
        w,h = img_sz
        test_image(str(outpath), h, w)
        # create_img_set(str(outpath),h=h,w=w)
        # if flip_wh:
            # create_img_set(str(outpath),h=w,w=h)

