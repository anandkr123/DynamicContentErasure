from sklearn.model_selection import train_test_split
from utils import *
import argparse
from multiresunet_secondmodel import MultiResUnet


desc = "MULTIRESUNET FOR DYNAMIC TEXT SEGMENTATION"
parser = argparse.ArgumentParser(description=desc)
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

args = parser.parse_args()

test_input_dir = args.test_input_dir
test_ground_dir = args.test_ground_truth_dir
test_batch_size = 1
img_ht = args.img_ht
img_wt = args.img_wt
img_ch = args.img_ch


# Reading of training and test images paths (function from utils.py)

total_input_path, total_ground_path, dataset_num = load_data(
    input_dir=join_dir(cwd, test_input_dir),
    ground_truth_dir=join_dir(cwd, test_ground_dir))


# Reading of test images for prediction
X_ = read_batch_images(dataset_num, total_input_path, img_ht, img_wt)
Y_ = read_ground_batch_images(dataset_num, total_ground_path, img_ht, img_wt)

model = MultiResUnet(height=img_ht, width=img_wt, n_channels=img_ch)


print("\n----------------Variables-------------------------")
print(f'\nTest data set size {dataset_num}')
print(f'\nImage size {img_ht} x {img_wt} x {img_ch}')
print(f'\nTest batch size {test_batch_size}')
print(f'\nTest input directory {test_input_dir}')
print(f'\nTest ground truth directory {test_ground_dir}')


print('\n------------TESTING and PREDICTING----------------')
print("\nRestoring model weights file from multiresunet_secondmodel_checkpoint, make sure it is present there\n")


# load model weights, should be the same file name when saved during training "e.g. multiresunet_big_tile.h5"
model.load_weights(join_dir(cwd, 'multiresunet_secondmodel_checkpoint', "multiresunet_delete.h5"))
test_mask = model.predict(X_, verbose=1, batch_size=test_batch_size)
test_loss = model.evaluate(X_, Y_, batch_size=test_batch_size)
print(f'\nTest loss {test_loss}')

#### SAVING PREDICTED IMAGES IN multiresunet_secondmodel_images directory
print("\n\n Creating directory to save predicted masked images --> multiresunet_secondmodel_images")
checkpoint_dir = check_folder('multiresunet_secondmodel_images')
print(f'\n-------Saving predcited images in multiresunet_secondmodel_images directory----------')
for i in range(dataset_num):
    rel_path = total_input_path[i].rsplit('/', 1)[1]
    rel_path = rel_path.replace('filled', 'predicted')
    predict = (test_mask[i])
    print(f'Single predicted image,  saving {rel_path}, shape is {predict.shape}')
    imsave(predict, join_dir(cwd, 'multiresunet_secondmodel_images', rel_path ))
    
    
    
    
    
    