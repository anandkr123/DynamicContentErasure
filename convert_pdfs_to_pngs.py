import argparse
import json
import math
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)


def str2bool(x):
    return x.lower() in ('true')


def save_image(img, img_name):
    cv2.imwrite(img_name, img)


def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def read_image(*args):
    img = cv2.imread(join_dir(*args), 0)
    return img


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def list_files(directory):
    files = os.listdir(join_dir(cwd, directory))
    files = filter(lambda x: not x.startswith('.'), files)
    return list(files)


def convert_single_pdf_png(open_filename, save_filename, des_directory=""):
    page = convert_from_path(open_filename, dpi=300)
    x = page[0]
    x.save(join_dir(cwd, des_directory, save_filename))


def convert_single_pdf_png_with_threshold(open_filename, save_filename, des_directory="", binary_inv=False):
    page = convert_from_path(open_filename, dpi=300)
    x = page[0]
    x.save(join_dir(cwd, des_directory, save_filename))
    img = read_image(join_dir(cwd, des_directory, save_filename))
    if binary_inv:
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 2)
    else:
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    save_image(th2, join_dir(cwd, des_directory, save_filename))


def convert_dir_pdf_png(pdf_dir, png_dir, threshold=False, binary_inv=False):
    check_dir(png_dir)
    counter_pngs = 0
    g_t = os.listdir(join_dir(cwd, pdf_dir))
    for a in sorted(g_t):
        if a.endswith('.pdf'):
            file_name = a.split('.')[0] + '.png'
            if threshold:
                counter_pngs += 1
                convert_single_pdf_png_with_threshold(join_dir(cwd, pdf_dir, a), file_name, png_dir, binary_inv)

                print("Converting {} to png with threshold, pdfs converted {}".format(a, counter_pngs))
            else:
                counter_pngs += 1
                convert_single_pdf_png(join_dir(cwd, pdf_dir, a), file_name, png_dir)

                print("Converting {} to png, pdfs converted {}".format(a, counter_pngs))

    return counter_pngs


def mask_dynamic_content(filled_thresh_dir, original_thresh_dir):
    check_dir('masked_fake_tax')
    print("\n Creating directory masked_fake_tax to store masked images\n")
    mask_counter = 0
    filled_thresh_paths = sorted(list_files(filled_thresh_dir))
    original_thresh_paths = sorted(list_files(original_thresh_dir))
    for filled_thresh_path, original_thresh_path in zip(filled_thresh_paths, original_thresh_paths):
        filled_image = read_image(join_dir(cwd, filled_thresh_dir, filled_thresh_path))
        original_image = read_image(join_dir(cwd, original_thresh_dir, original_thresh_path))
        mask = original_image - filled_image
        save_image(mask, join_dir(cwd, 'masked_png', filled_thresh_path))
        mask_counter += 1
        print("\nSaving mask image of {}, Masking {} image ".format(filled_thresh_path, mask_counter))
    return mask_counter


desc = "Convert pdfs in directory to pngs"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-f', '--pdf_dir_first', type=str, default='filled_pdf_fake_tax',
                    help='Directory of the first pdf files')
parser.add_argument('-s', '--pdf_dir_second', type=str, default='original_pdf_fake_tax',
                    help='Directory of the second pdf files')

parser.add_argument('-e', '--png_dir_first', type=str, default='filled_png_fake_tax',
                    help='Directory where to save the png files')
parser.add_argument('-d', '--png_dir_second', type=str, default='original_png_fake_tax',
                    help='Directory where to save the png files')

parser.add_argument('-t', '--thresh', type=str, default="False", help='whether to do threshold after pdf to png')

parser.add_argument('-b', '--binary_inv', type=str, default="False", help='Binary image or its inverse')

args = parser.parse_args()

if args.pdf_dir_first and args.pdf_dir_second:

    total_converted_filled = convert_dir_pdf_png(args.pdf_dir_first, args.png_dir_first, threshold=False,
                                                 binary_inv=str2bool(args.binary_inv))

    total_converted_filled_thresh = convert_dir_pdf_png(args.pdf_dir_first, 'filled_png_thresh_fake_tax', threshold=True,
                                                        binary_inv=str2bool(args.binary_inv))

    total_converted_original = convert_dir_pdf_png(args.pdf_dir_second, args.png_dir_second, threshold=False,
                                                   binary_inv=str2bool(args.binary_inv))

    total_converted_original_thresh = convert_dir_pdf_png(args.pdf_dir_second,
                                                          'original_png_thresh_fake_tax', True, str2bool(args.binary_inv))

    total_masked_images = mask_dynamic_content('filled_png_thresh_fake_tax',
                                               'original_png_thresh_fake_tax')

    if total_converted_filled == total_converted_original == total_masked_images:
        print("Converted {} filled and original pdf's to png successfully and also created mask of it".format(
            total_converted_original))
else:
    exit()
