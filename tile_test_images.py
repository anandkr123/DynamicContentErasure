import argparse
import glob
import json
import math
import sys

import cv2
import numpy as np
from pdf2image import convert_from_path
import os
from more_itertools import divide
import matplotlib.pyplot as plt
import time
import shutil

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)


def list_files(directory):
    files = os.listdir(join_dir(cwd, directory))
    files = filter(lambda x: not x.startswith('.'), files)
    return list(files)


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def read_image(*args):
    img = cv2.imread(join_dir(*args), 0)
    return img


def tile_image_no_overlap(image_path, tile_size_height, tile_size_width, pixels_reduce=None, padding_value=0, directory=""):
    """

    :param image_path: path of the image to be tiled
    :param tile_size_height: height of the tile
    :param tile_size_width: width of the tile
    :param pixels_reduce: 1D array of 4 value to specify the offset in the image
    :param padding_value: pad the image with this value
    :param directory: directory of the image
    :return: None
    """

    dict_tiled_image_path = {}
    total_tiled_images = []
    tile_images_counter = 0
    total_predicted_tiled_images = []

    img = read_image(cwd, directory, image_path)

    img_rows, img_cols = img.shape[0], img.shape[1]
    if pixels_reduce:
        row_reduce_top, col_reduce_left, row_reduce_bottom, col_reduce_right = pixels_reduce[0], pixels_reduce[1], \
                                                                               pixels_reduce[2], pixels_reduce[3]
        img = img[row_reduce_top: img_rows - row_reduce_bottom, col_reduce_left: img_cols - col_reduce_right]

    modified_rows, modified_cols = img.shape[0], img.shape[1]

    # change to math.ceil to pad the rows and columns and adjust the padding_value parameter accordingly
    numrows, numcols = math.ceil(modified_rows / tile_size_height), math.ceil(modified_cols / tile_size_width)

    # code to append padding borders in image
    top = 0
    left = 0
    right = numcols*tile_size_width - modified_cols
    bottom = numrows*tile_size_height - modified_rows

    # value to be 0 in case of ground truth images and 255 in case of filled images
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)

    height = tile_size_height
    width = tile_size_width
    for row in range(numrows):
        for col in range(numcols):
            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width
            img_crp = img[y0:y1, x0:x1]
            tiled_filled_dir = "tiled_realistic_filled_test_images"
            tiled_masked_dir = "tiled_realistic_masked_test_images"
            filename = image_path.split('.')[0]
            tiled_imaged_name = "{}_{}_{}.png".format(filename, str(row).zfill(2), str(col).zfill(2))
            if 'filled' in image_path:
                check_dir(tiled_filled_dir)
                cv2.imwrite(join_dir(cwd, tiled_filled_dir, tiled_imaged_name), img_crp)
                total_tiled_images.append(join_dir(cwd, tiled_filled_dir, tiled_imaged_name))
                tile_images_counter += 1
            else:
                check_dir(tiled_masked_dir)
                cv2.imwrite((join_dir(cwd, tiled_masked_dir, tiled_imaged_name)), img_crp)
                total_tiled_images.append(join_dir(cwd, tiled_masked_dir, tiled_imaged_name))
                tile_images_counter += 1

                pred_tiled_image_name = tiled_imaged_name.replace('masked', 'predicted')
                total_predicted_tiled_images.append(
                    join_dir(cwd, 'tiled_realistic_predicted_masked_images', pred_tiled_image_name))

    dict_tiled_image_path[image_path] = total_tiled_images

    if 'masked' in image_path:
        pred_image = image_path.replace('masked', 'predicted')
        dict_tiled_image_path[pred_image] = total_predicted_tiled_images

    if os.path.exists(join_dir(cwd, 'tiled_image_paths.json')):
        with open('tiled_image_paths.json') as f:
            data = json.load(f)

        data.update(dict_tiled_image_path)

        with open('tiled_image_paths.json', 'w') as f:
            json.dump(data, f)
    else:
        with open('tiled_image_paths.json', mode='w') as f:
            json.dump(dict_tiled_image_path, f, indent=2)

    print("\nWritten {} images successfully".format(tile_images_counter))


desc = "Tile test images with no overlap"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--filled_test_png_dir', type=str, default='realistic_filled_test_images',
                    help='Directory of filled png test images')
parser.add_argument('--masked_test_png_dir', type=str, default='realistic_masked_test_images',
                    help='Directory of masked png test images')
parser.add_argument('--tile_img_ht', type=int, default=512,
                    help='height of tiled image')
parser.add_argument('--tile_img_wt', type=int, default=512,
                    help='width of tiled image')

args = parser.parse_args()

print("\n Creating tiled_realistic_filled_test_images and tiled_realistic_masked_test_images "
      "directories to save the tiled test images\n ")

total_images_tiled = 0

if args.filled_test_png_dir and args.masked_test_png_dir:

    for file_i, file_o in zip(sorted(list_files(args.filled_test_png_dir)), sorted(list_files(args.masked_test_png_dir))):
        total_images_tiled += 2
        tile_image_no_overlap(file_i, args.tile_img_ht, args.tile_img_wt, pixels_reduce=None, padding_value=255,
                              directory=args.filled_test_png_dir)
        tile_image_no_overlap(file_o, args.tile_img_ht, args.tile_img_wt, pixels_reduce=None, padding_value=0,
                              directory=args.masked_test_png_dir)

        print(f'\n Saved tiled IMAGES path of filled, masked and predicted in JSON file tiled_image_paths.json')
        print(f'\nTiled {total_images_tiled//2} pair of filled and masked ground truth test images successfully\n\n')
else:
    sys.exit("Specify the filled and masked png test directories to tile")

