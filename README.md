# DynamicContentErasure

Dynamic content erasure that learns to differentiate between dynamic and static text present in a document image and selectively finds and erases dynamic content present in a filled form document.

Model flow at training time

[training_model.pdf](https://github.com/anandkr123/DynamicContentErasure/files/6322903/training_model.pdf)


Model flow during inference mode

[testing_model.pdf](https://github.com/anandkr123/DynamicContentErasure/files/6322904/testing_model.pdf)


Dice score on 18 test forms

Dice score between target mask and restore masked image

<img width="683" alt="dice_score_dynamic_content" src="https://user-images.githubusercontent.com/23450113/114979647-8e4d3780-9e8b-11eb-8cdb-8ad7e99b41fb.png">

Dice score between original empty form and restore original form

<img width="657" alt="dice_score_static_content" src="https://user-images.githubusercontent.com/23450113/114979651-8f7e6480-9e8b-11eb-8867-20a1ff74301a.png">
