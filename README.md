# DynamicContentErasure

Dynamic content erasure that learns to differentiate between dynamic and static text present in a document image and selectively finds and erases dynamic content present in a filled form document.

## Model flow at training time

![training_model](https://user-images.githubusercontent.com/23450113/221436039-9daf6a9a-c393-40c7-8d52-8bb0959b0374.png)


## Model flow during inference mode

![testing_model-1](https://user-images.githubusercontent.com/23450113/221436069-6556c934-f29f-429b-8644-d3aa99c34ed1.png)



## Directories description

**"filled_png_test_thresh"**                    --> 18 test forms to evaluate the results.

**"realistic_masked_test_images"**              --> Target masked form images.

**"realistic_threshold_predicted_test_images"** --> The predicted masked results on 18 test forms.

**"predicted_original_form"**                   --> Restored empty original form obtained from subtracting "filled_png_test_thresh" from       "realistic_threshold_predicted_test_images"             


## Scripts description

**generate_pdf_dataset.py**      --> Extract relevant pdfs from the forms fulfilling the pdfs selection criteria and populate it with text at random coordinates.

**convert_pdfs_to_pngs.py**     --> Converts pdfs to pngs and generate masked document images (target mask) of dynamic content.

**tile_train_images.py**         --> Tiling the input dataset to feed into the neural network.

**tile_test_images.py**        -> Tiling the test images to predict the results. Stores the tiled image path in JSON file to remerge it and evaluate the results.

**merge_masked_and cal_dice_score.py**  --> Merged the predicted masked images and caluclate the dice score between target masked images and merged masked images.

**restore_original_and_cal_dice_score.py** --> Restore the predicted original empty form using image processing and calculate the dice score.

**fake_data_generator.py**       --> Generates the fake tax data set

**multiresunet_secondmodel.py**  --> MutiResUnet model architecture (link- https://arxiv.org/pdf/1902.04049.pdf)

**multiresunet_secondmodel_train.py**   --> Read input images using a generator function, trains the model using GPU.

**multiresunet_secondmodel_test.py**    --> Reads test images and predicts the results from the pre-trained model


## Dice score on 18 test forms

Dice score between target mask and restore masked image

<img width="683" alt="dice_score_dynamic_content" src="https://user-images.githubusercontent.com/23450113/114979647-8e4d3780-9e8b-11eb-8cdb-8ad7e99b41fb.png">

## Dice score between original empty form and restore original form

<img width="657" alt="dice_score_static_content" src="https://user-images.githubusercontent.com/23450113/114979651-8f7e6480-9e8b-11eb-8867-20a1ff74301a.png">
