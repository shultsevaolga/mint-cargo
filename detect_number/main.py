import argparse
import cv2
import numpy as np
import pytesseract 
from matplotlib import pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input image path")
    parser.add_argument("--min-y", type=int, default=0, help="min y carriage coordinate")
    parser.add_argument("--max-y", type=int, default=-1, help="max y carriage coordinate")
    parser.add_argument("--number-ratio", type=float, default=4.5, help="number width/height ratio")
    return parser.parse_args()


def detect_number(img, min_y, max_y, number_ratio):
    img_width = img.shape[1]
    cropped_img = img[min_y:max_y, :, :]

    # add some contrast
    contrast_img = cv2.convertScaleAbs(cropped_img, alpha=2, beta=1)

    # make image gray and add some blur
    gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (5, 5), 1)

    # make adaptive threshold for better edges
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

    # add some erode to make contours bigger
    img_erode = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=2)

    # find contours
    (contours, _) = cv2.findContours(img_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for i, contour in enumerate(contours):
        # make contour convex hull and find its bounding rectangle
        eps = 0.25
        hull = cv2.convexHull(contour, False)
        bounding_rect = cv2.boundingRect(hull)
        hull_width = bounding_rect[2]

        # Some contour checks, need to refine
        if hull_width < img_width / 3. or hull_width > img_width * 0.95:
            continue

        if np.abs(number_ratio / bounding_rect[2] * bounding_rect[3] - 1) < eps:
            break

    padding = 10
    nmin_y = bounding_rect[1] + padding
    nmax_y = bounding_rect[1] + bounding_rect[3] - padding
    nmin_x = bounding_rect[0] + padding
    nmax_x = bounding_rect[0] + bounding_rect[2] - padding

    # crop to number area
    number_img = cropped_img[nmin_y:nmax_y, nmin_x:nmax_x, :]

    # make gray and add some blur
    gray_number = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
    blur_number = cv2.blur(gray_number, (5, 5), 1)

    # calculate threshold and scale
    _, number_thresh = cv2.threshold(blur_number, 200, 255, cv2.THRESH_OTSU)
    number_scaled = cv2.convertScaleAbs(number_thresh, alpha=2, beta=1)

    # erode and add some borders
    number_eroded = cv2.erode(number_scaled, np.ones((2, 2), np.uint8), iterations=3)
    number_bordered = cv2.copyMakeBorder(number_eroded, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return pytesseract.image_to_string(number_bordered, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')


def main():
    params = parse_arguments()
    img = cv2.imread(params.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        number_text = detect_number(img, params.min_y, params.max_y, params.number_ratio)
        print(f"Carriage number: {number_text}")
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
