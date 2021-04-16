import argparse
import glob
import json
import math
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import sys
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
    tile_images_counter = 0
    img = read_image(cwd, directory, image_path)
    img_rows, img_cols = img.shape[0], img.shape[1]
    if pixels_reduce:
        row_reduce_top, col_reduce_left, row_reduce_bottom, col_reduce_right = pixels_reduce[0], pixels_reduce[1], \
                                                                               pixels_reduce[2], pixels_reduce[3]
        img = img[row_reduce_top: img_rows - row_reduce_bottom, col_reduce_left: img_cols - col_reduce_right]

    modified_rows, modified_cols = img.shape[0], img.shape[1]

    # change to math.ceil to pad the rows and columns and adjust the padding_value parameter accordingly
    numrows, numcols = math.floor(modified_rows / tile_size_height), math.floor(modified_cols / tile_size_width)

    # code to append padding borders in image
    # top = 0
    # left = 0
    # right = numcols*tile_size_cols - modified_cols
    # bottom = numrows*tile_size_rows - modified_rows
    # # value to be zero in case of ground truth images  when we want the background pixels as black
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)

    height = tile_size_height
    width = tile_size_width
    for row in range(numrows):
        for col in range(numcols):
            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width
            img_crp = img[y0:y1, x0:x1]
            tiled_filled_dir = "tiled_filled_png_fake_tax"
            tiled_masked_dir = "tiled_masked_png_fake_tax"
            filename = image_path.split('.')[0]
            tiled_imaged_name = "{}_{}_{}.png".format(filename, str(row).zfill(2), str(col).zfill(2))
            if 'filled' in image_path:
                check_dir(tiled_filled_dir)
                cv2.imwrite(join_dir(cwd, tiled_filled_dir, tiled_imaged_name), img_crp)
                tile_images_counter += 1
            else:
                check_dir(tiled_masked_dir)
                cv2.imwrite((join_dir(cwd, tiled_masked_dir, tiled_imaged_name)), img_crp)
                tile_images_counter += 1

    print("Written {} images successfully".format(tile_images_counter))


desc = "Tile train images with no overlap"
total_unique_forms = 274
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--filled_png_dir', type=str, default='filled_png_fake_tax',
                    help='Directory of filled png train image')
parser.add_argument('--masked_png_dir', type=str, default='masked_fake_tax',
                    help='Directory of masked png train images')
parser.add_argument('--tile_img_ht', type=int, default=512,
                    help='height of tiled image')
parser.add_argument('--tile_img_wt', type=int, default=512,
                    help='width of tiled image')
parser.add_argument('--form_var', type=int, default=2,
                    help='Number of single form image variations to tile')

args = parser.parse_args()
print("\n Creating tiled_filled_png_fake_tax and tiled_masked_fake_tax directories to save the tiled images\n ")

if args.filled_png_dir and args.masked_png_dir:
    for i in range(total_unique_forms):
        var_tile_counter = 1  # TILE ONLY DEFINED NO. OF IMAGES
        var_counter = str(i + 1).zfill(4)
        prefixed_filled = [filename for filename in sorted(os.listdir(args.filled_png_dir)) if
                           filename.startswith(var_counter)]
        prefixed_original = [filename for filename in sorted(os.listdir(args.masked_png_dir)) if
                             filename.startswith(var_counter)]
        for file_i, file_o in zip(sorted(prefixed_filled), sorted(prefixed_original)):
            if var_tile_counter > args.form_var:
                break
            else:
                tile_image_no_overlap(file_i, args.tile_img_ht, args.tile_img_wt, padding_value=255,
                                      directory=args.filled_png_dir)
                tile_image_no_overlap(file_o, args.tile_img_ht, args.tile_img_wt, padding_value=0,
                                      directory=args.masked_png_dir)
                var_tile_counter += 1
            print(f'Tiled {var_tile_counter - 1}  variations of filled and masked image {var_counter} ')
        print("----------------Tiled {} image of its filled and masked variations------------- ".format(var_counter))
else:
    sys.exit("Specify the filled and masked png directories to tile")
