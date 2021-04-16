import json
import math
import cv2
import numpy as np
import argparse
import os
from more_itertools import divide
import matplotlib.pyplot as plt

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)

desc = "Restore original form and calculate dice score"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--pred_orig_form_dir', type=str, default="predicted_original_form",
                    help='Directory to save the restored original form')

args = parser.parse_args()
predicted_original_form_directory = args.pred_orig_form_dir

if not args.pred_orig_form_dir:
    exit()


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


def save_original_form(filled_image_name, predicted_masked_image_name, filled_dir, masked_dir):
    """
    both of the image must be already binary images to get clear results

    :param filled_image_name: filled image name
    :param predicted_masked_image_name: predicted image name
    :param filled_dir: directory of filled test image
    :param masked_dir: directory of predicted masked images
    :return: saves the restored original form in "predicted_original_form" directory
    """

    filled_image = read_image(join_dir(cwd, filled_dir, filled_image_name))
    predicted_masked_image = read_image(join_dir(cwd, masked_dir, predicted_masked_image_name))
    predicted_masked_image = 255 - predicted_masked_image
    inv_image = predicted_masked_image - filled_image
    original_form = 255 - inv_image
    check_dir(join_dir(cwd, predicted_original_form_directory))
    filled_image_name = filled_image_name.replace('filled', 'original')
    cv2.imwrite(join_dir(cwd, predicted_original_form_directory, filled_image_name), original_form)
    print(f'\nSaving predicted {filled_image_name} in predicted_original_form directory ')


print("\n Filled png test thresh directory                   -->   filled_png_test_thresh")

print("\n Thresholded masked predicted test images directory -->  realistic_threshold_predicted_test_images")

print("\n Original png test thresh directory                 -->  original_png_test_thresh ")


# RESTORE ORIGINAL FORM
print("\n Creating predicted_original_form directory to save the restored original form")
for filled_name, predicted_name in zip(sorted(list_files('filled_png_test_thresh')),
                                       sorted(list_files('realistic_threshold_predicted_test_images'))):

    save_original_form(filled_name, predicted_name, 'filled_png_test_thresh',
                       'realistic_threshold_predicted_test_images')


#  CALCULATING THE DICE SCORE
sub_dir = ""
original_png_thresh_dir = "original_png_test_thresh"

# already binarized
original = sorted(list_files(join_dir(sub_dir,
                                      original_png_thresh_dir)) )
# already binarized
predicted = sorted(list_files(join_dir(sub_dir,
                                       predicted_original_form_directory)))
avg_dice_score = []
dice_score = 0

for pred_path, truth_path in zip(predicted, original):

    truth_image = read_image(join_dir(cwd, sub_dir,
                                      original_png_thresh_dir, truth_path))
    pred_image = read_image(join_dir(cwd, sub_dir,
                                     predicted_original_form_directory, pred_path))

    # THE dice score is only for binary images
    # careful if the image is already scaled between 0-1
    if pred_image.dtype == 'uint8' and truth_image.dtype == 'uint8':

        # Dice score over black pixels (static content)
        pred_image = 255 - pred_image
        truth_image = 255 - truth_image

        # now only 0 or 1
        # converting the data type to float64

        pred_image = pred_image / 255
        truth_image = truth_image / 255

        if pred_image.shape == truth_image.shape:
            dice_score = single_dice_coef(truth_image, pred_image)
        else:
            print('Predicted image and ground truth image mis-match in dimensions')
            exit()
        print(f'\nDice score between {pred_path} and {truth_path} is {dice_score}')
        avg_dice_score.append(dice_score)
    else:
        _, pred_image_thresh = cv2.threshold(pred_image, 0.3, 1, cv2.THRESH_BINARY)
        print("Copy the above code of if block carefully and calc. the dice score for it")

print(f'The average dice score is {np.mean(avg_dice_score)}')
