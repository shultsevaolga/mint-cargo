import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input image path")
    parser.add_argument("-o", "--output", default="out.png", help="output image path")
    parser.add_argument("--min-y", type=int, default=0, help="min y carriage coordinate")
    parser.add_argument("--max-y", type=int, default=-1, help="max y carriage coordinate")
    return parser.parse_args()


def process_img(img, min_y, max_y):
    # adapt image to our case
    cropped_img = img[min_y:max_y, :]

    # make image more contrast
    contrast_img = cv2.convertScaleAbs(cropped_img, alpha=1.2, beta=10)

    # convert to grayscale
    gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)

    # add some blur for better edges
    blur = cv2.blur(gray, (20, 20), 2)

    # find edges
    canny = cv2.Canny(blur, 10, 50)
    (y, x) = np.where(canny == 255)

    if not x.any() or not y.any():
        raise RuntimeError("Cannot process image")

    # calculate boundaries
    padding = 20
    top_y, top_x = np.min(y), np.min(x)
    bottom_y, bottom_x = np.max(y), np.max(x)
    carriage = cropped_img[top_y+padding:bottom_y-padding, top_x+padding:bottom_x-padding]
    contrast_carriage = cv2.convertScaleAbs(carriage, alpha=1.2, beta=10)

    window_size = 15
    threshold_norm = 90
    hh = contrast_carriage.shape[0]
    ww = contrast_carriage.shape[1]
    mask = np.uint8(np.zeros(contrast_carriage.shape[:2]))
    for y in range(0, hh, window_size):
        for x in range(0, ww, window_size):
            avg_color = np.average(contrast_carriage[y:y+window_size, x:x+window_size, :], axis=(0, 1))
            mask[y:y+window_size, x:x+window_size] = (np.linalg.norm(avg_color) < threshold_norm)

    percent = np.sum(mask) / (contrast_carriage.shape[0] * contrast_carriage.shape[1])
    _, threshold_pepper = cv2.threshold(255*mask, 1, 1, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold_pepper, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(carriage, contours, -1, 255, 3)
    img[
        top_y+min_y+padding:bottom_y+min_y-padding,
        top_x+padding:bottom_x-padding
    ] = carriage
    return percent, img


def main():
    params = parse_arguments()
    img = cv2.imread(params.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        percent, processed_img = process_img(img, params.min_y, params.max_y)
    except Exception as e:
        print(f"Processing failed: {e}")
        percent = 0
        processed_img = img
    plt.imsave(params.output, processed_img)
    print(f"Pepper percent: {percent}")


if __name__ == "__main__":
    main()
