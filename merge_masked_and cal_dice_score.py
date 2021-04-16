import json
import math
import sys
import argparse
import cv2
import numpy as np
import os
from more_itertools import divide
import matplotlib.pyplot as plt

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)

desc = "Merge tiled predicted masked images and calculate dice score"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--json_file', type=str, default='tiled_image_paths.json',
                    help='Name of the JSON file')
args = parser.parse_args()

if not args.json_file:
    sys.exit("No json file present! add the json file")


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


def single_dice_coef(y_true, y_pred_bin):
    """

    :param y_true: ground truth image
    :param y_pred_bin: predicted image
    :return: dice score value 0-1
    """
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
        return 1
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin))


def tiled_images_paths(total_h_tiles, data, filename):
    list_of_images_path = data[filename]
    ordered_image_paths = [divide(total_h_tiles, list_of_images_path)]
    return ordered_image_paths


def merge_images(filename, json_file, directory=""):
    """

    :param filename: image filename to be merged . e.g. predicted_0001.png
    :param json_file: json file which contains path info. of merged tiles
    :param directory: directory where the json file is. by default its cwd
    :return: the merged image
    """
    # JSON file contains the complete path of the tiled images

    # takes the directory name where JSON file resides, by default looks in current directory
    with open(join_dir(cwd, directory, json_file), 'r') as f:
        data = json.load(f)

    tiled_image_paths = data[filename]
    total_tiles = len(tiled_image_paths)
    single_tile = tiled_image_paths[-1]

    # splitting the tiled image path name '.png'
    last_tiled_file = single_tile.rsplit('/', 1)[1]
    last_tiled_file = last_tiled_file.split('.')[0]

    # predicted_0012_06_04 (06 is the total no. of horizontal tiles)
    total_h_tiles = (last_tiled_file.split('_')[2]).strip('0')

    # adding 1, as it starts from 0
    total_h_tiles = int(total_h_tiles) + 1

    total_v_tiles = total_tiles // total_h_tiles

    # rearranges the tiled images path in the order of merging
    ordered_paths_iter_obj = tiled_images_paths(total_h_tiles, data, filename)

    # reading a single tile to get the dimensions of the tile
    image_tile = read_image(single_tile)
    r_t, c_t = image_tile.shape[0], image_tile.shape[1]

    # a black horizontal tile to start with
    h_tile = np.zeros(shape=(r_t, c_t), dtype=np.uint8)

    # a black vertical tile to start with
    v_tile = np.zeros(shape=(r_t, c_t + c_t * total_v_tiles), dtype=np.uint8)

    for h_path_it in ordered_paths_iter_obj:
        for h_paths in h_path_it:
            h_tile_list = list(h_paths)
            for img_path in h_tile_list:
                img = read_image(img_path)
                h_tile = np.concatenate((h_tile, img), axis=1)
            v_tile = np.concatenate((v_tile, h_tile), axis=0)
            h_tile = np.zeros(shape=(r_t, c_t), dtype=np.uint8)

    tile = v_tile[r_t: r_t * (total_h_tiles + 1), c_t: c_t * (total_v_tiles + 1)]

    original_made = tile[0:3508, 0:2481]

    return original_made


# the directory name helps in numbering of files
directory_name = "realistic_masked_test_images"
print(f'\nCreating directory realistic_predicted_test_images to store the merged images')
for filename in sorted(list_files(join_dir(cwd, directory_name))):
    filename = filename.replace('masked', 'predicted')
    image = merge_images(filename, json_file=args.json_file)
    check_dir(join_dir(cwd, 'realistic_predicted_test_images'))
    cv2.imwrite(join_dir(cwd, 'realistic_predicted_test_images', filename), image)
    print(f'\nSuccessfully merged {filename} in realistic_predicted_test_images directory')


# calculating the dice score
original = sorted(list_files('realistic_masked_test_images'))
predicted = sorted(list_files('realistic_predicted_test_images'))  ### not yet binarized,
avg_dice_score = []
dice_score = 0


print("\n ORIGINAL MASKED png test images directory -->   realistic_masked_test_images")

print(f'\n\nCreating directory realistic_threshold_predicted_test_images directory to store the threshold merged images')

for pred_path, truth_path in zip(predicted, original):
    truth_image = read_image(join_dir(cwd, 'realistic_masked_test_images', truth_path))
    pred_image = read_image(join_dir(cwd, 'realistic_predicted_test_images', pred_path))

    # THE dice score is only for binary images
    # careful if the image is already scaled between 0-1
    if pred_image.dtype == 'uint8' and truth_image.dtype == 'uint8':

        # first threshold, only 0 or 255, 70 is a guessed threshold value taken by visualisation and expertise
        _, pred_image_thresh = cv2.threshold(pred_image, 70, 255, cv2.THRESH_BINARY)

        # now only between 0 or 1
        pred_image_thresh = pred_image_thresh / 255
        truth_image = truth_image / 255
        check_dir(join_dir(cwd, 'realistic_threshold_predicted_test_images'))

        # multiply by 255 as open cv saves in uint8 format
        cv2.imwrite(join_dir(cwd, 'realistic_threshold_predicted_test_images', pred_path), pred_image_thresh * 255)
        if pred_image.shape == truth_image.shape:
            dice_score = single_dice_coef(truth_image, pred_image_thresh)
        else:
            print('Predicted image and ground truth image mis-match in dimensions')
            exit()
        print(f'\nDice score between {pred_path} and {truth_path} is {dice_score}')
        avg_dice_score.append(dice_score)
    else:
        _, pred_image_thresh = cv2.threshold(pred_image, 0.3, 1, cv2.THRESH_BINARY)
        print("Copy the above code of if block carefully and calc. the dice score for it")

print(f'The average dice score is {np.mean(avg_dice_score)}')
