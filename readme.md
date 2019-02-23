## Homework 1 for COMP 7950: Deep Learning

### 
This structure of the project is forked from "Project Starter Package" of CS230 course
from Stanford.

link: https://cs230-stanford.github.io/

github repo: https://github.com/cs230-stanford/cs230-code-examples

### How to run the code at any unix machine
For this assignment only.

Step 1 : Create virtual env using pip and the available requirement.txt (in case of local setup)

Step 2 : Make sure the 'raw_image_dir' class variable points to appropriate image folder. In case you are running
the code in any linux machine use '/import/helium-share/staff/ywang/comp7950/images'.

Step 3 : Create dataset by executing build_dataset.py. this class loads the training dataset into train.txt, val.txt (80:20 split)
and the test dataset into test.txt. (each of these file contains filenames and labels only, no raw numpy image data). This branch
already have the split so execute this only if you need to shuffle the train.txt and val.txt.

python build_dataset.py --data_dir /import/helium-share/staff/ywang/comp7950 --output_dir data/localization_dataset


Step 4: Train and cross validate the model by executing train_localize.py which will use hyper parameters from params.json found in the 'model_dir' 

python train_localize.py --data_dir data/localization_dataset --model_dir experiments/base_model


Step 5: The test we don't have any labels and the assignment asked for writing the  output values to txt file for submission 
which will use hyper parameters from params.json found in the 'model_dir' and restore the 'best model' found in the 'model_dir'

python test_localize.py --data_dir data/localization_dataset --model_dir experiments/base_model


### Submission specific details

* #### Hyper parameters
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 1000,
    "num_channels": 224,
    "save_summary_steps": 100,
    "num_workers": 4
}
* used pre-trained Resnet18 with modified last layer to fit the regression problem. I have tested two cases:
    * only train the last (customized) layer, which gave around 52-55% training accuracy.
    * training the whole network which gave result aligned with assignment description (around 77% after 10 epoch.)
* The submitted weights are for the second case where the whole network has been trained.





