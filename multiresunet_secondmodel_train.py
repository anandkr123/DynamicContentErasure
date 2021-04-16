import sys
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from multiresunet_secondmodel import MultiResUnet
from utils import *
from random import randrange

desc = "MULTIRESUNET FOR DYNAMIC TEXT SEGMENTATION"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--train_input_dir', type=str, default='tiled_filled_png_fake_tax',
                    help='Directory of input images')

parser.add_argument('--train_ground_truth_dir', type=str, default='tiled_masked_fake_tax',
                    help='Directory of ground-truth masked images')

parser.add_argument('--test_input_dir', type=str, default='tiled_realistic_filled_test_images',
                    help='Directory of test input images')

parser.add_argument('--test_ground_truth_dir', type=str, default='tiled_realistic_masked_test_images',
                    help='Directory of test ground-truth masked images')

parser.add_argument('--img_ht', type=int, default=512,
                    help='Image height')

parser.add_argument('--img_wt', type=int, default=512,
                    help='Image width')

parser.add_argument('--img_ch', type=int, default=1,
                    help='Image channels')

parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to run')

parser.add_argument('--batch_size', type=int, default=2, 
                    help='batch_size')

parser.add_argument('--single_gpu', type=str, default="True",
                    help='Use multi or single gpu')

args = parser.parse_args()

validation_data_fraction = 0.999
train_batch_size = args.batch_size
img_ht = args.img_ht
img_wt = args.img_wt
img_ch = args.img_ch
epochs = args.epochs

# Reading of training and test images paths (function from utils.py)

total_input_path, total_ground_path, dataset_num = load_data(
    input_dir=join_dir(cwd, args.train_input_dir),
    ground_truth_dir=join_dir(cwd, args.train_ground_truth_dir))

test_input_path, test_ground_path, test_dataset_num = load_data(
    input_dir=join_dir(cwd, args.test_input_dir),
    ground_truth_dir=join_dir(cwd, args.test_ground_truth_dir))

# exclude some training data if required, set validation_data_fraction = very low if almost all training data is needed.
input_path, test, ground_path, test_ground = train_test_split(
    total_input_path, total_ground_path,
    test_size=validation_data_fraction, random_state=4)

# shuffling of data
input_path, ground_path = shuffle(input_path, ground_path, random_state=randrange(100))

# reading of test images, used for early stopping
include_test_input_path, exclude_test_input_path, include_test_ground_path, exclude_test_ground_path = train_test_split(
    test_input_path, test_ground_path,
    test_size=0.01, random_state=4)


# A generator function for infinite iteration of training images

def train_data_generator(input_path, ground_path, batch_size=train_batch_size):
    while True:

        input_path, ground_path = shuffle(input_path, ground_path, random_state=randrange(100))

        for i, j in zip(range(len(input_path) // batch_size), range(len(ground_path) // batch_size)):
            yield (read_batch_images(batch_size, input_path[i * batch_size: (i + 1) * batch_size], img_ht, img_wt),
                   read_ground_batch_images(batch_size, ground_path[j * batch_size: (j + 1) * batch_size], img_ht,
                                            img_wt))

# a gen expression             
gen = train_data_generator(input_path, ground_path, batch_size=train_batch_size)


# X_ = read_batch_images(dataset_num, total_input_path, img_ht, img_wt)
# Y_ = read_ground_batch_images(dataset_num, total_ground_path, img_ht, img_wt)

# X_train = read_batch_images(train_data_size, input_path, img_ht, img_wt)
# Y_train = read_ground_batch_images(train_data_size, ground_path, img_ht, img_wt)

# X_test = read_batch_images(test_dataset_num, test_input_path, img_ht, img_wt)
# Y_test = read_ground_batch_images(test_dataset_num, test_ground_path, img_ht, img_wt)


#### fraction of input and test images
# X_ = read_batch_images(len(input_path), input_path, img_ht, img_wt)
# Y_ = read_ground_batch_images(len(input_path), ground_path, img_ht, img_wt)

# reads all test images at once to test after each epoch
X_test = read_batch_images(len(include_test_input_path), include_test_input_path, img_ht, img_wt)
Y_test = read_ground_batch_images(len(include_test_ground_path), include_test_ground_path, img_ht, img_wt)


### SAVING THE BEST MODEL ####

checkpoint_dir = check_folder('multiresunet_secondmodel_checkpoint')
checkpoint_path = join_dir(cwd, 'multiresunet_secondmodel_checkpoint', "multiresunet_delete.h5")
ckpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

## EARLY STOPPING BASED ON VALIDATION LOSS ###

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=12,
    verbose=0,
    mode='min',
    baseline=None,
    restore_best_weights=True
)

callbacks_list = [ckpt, early_stopping_monitor]

print("\n----------------Variables-------------------------")
print("\n Train Input directory ----{args.train_input_dir}")
print("\n Train Groundtruth directory ----{args.train_ground_truth_dir}")
print(f'\nTrain dataset size ------ {len(input_path)}')
print(f'\nImage size -------------- {img_ht} x {img_wt} x {img_ch}')
print(f'\nValidation dataset size-- {len(include_test_ground_path)}')
print(f'\nBatch size -------------- {train_batch_size}')
print(f'\nEpochs------------------- {epochs}')

print("\n\nCreating the model checkpoint folder --> multiresunet_secondmodel_checkpoint")

if epochs >=1:
    if args.single_gpu == "True":
        print("\n ------------Training on single gpu--------------")
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        with tf.device("/gpu:1"):
            u_net = MultiResUnet(height=img_ht, width=img_wt, n_channels=img_ch)
            u_net.fit(gen, validation_data=(X_test, Y_test), epochs=epochs, steps_per_epoch=len(input_path) // train_batch_size,
              callbacks=callbacks_list)
    else:
        print("\n----------Training on multi-gpus------------------")
        strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])
        with strategy.scope():
            u_net = MultiResUnet(height=img_ht, width=img_wt, n_channels=img_ch)
            u_net.fit(gen, validation_data=(X_test, Y_test), epochs=epochs, steps_per_epoch=len(input_path) // train_batch_size,
              callbacks=callbacks_list)
else:
    sys.exit("Number of epochs must be greater than or equal to 1")


print("--------TRAINING FINISHED----------")


# tf.debugging.set_log_device_placement(False)

# TRAINING on 2 GPU's

    
# Uncomment to train on single GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# with tf.device("/gpu:1"):
#     u_net = MultiResUnet(height=img_ht, width=img_wt, n_channels=img_ch)
#     u_net.fit(gen, validation_data=(X_test, Y_test), epochs=epochs, steps_per_epoch=len(input_path) // train_batch_size,
#               callbacks=callbacks_list)
#



