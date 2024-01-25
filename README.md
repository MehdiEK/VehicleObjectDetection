# Vehicle Object Detection - VIC challenge 2024

In order to have more details on the general methodology, please check the report. 

Creation date: XX/XX/2024
Last modification: 23/01/2024
By: Mehdi EL KANSOULI 


## Intrduction 

The objective of this work is to detect vehicles in images. To achieve this, we proceeded in three steps: external data collection, training of a binary classifier (vehicle or not), and construction of a pipeline based on a method known as "sliding window."


## External data folder
We began by searching for a dataset to train a classifier specifically for this task (\url{https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set}). This balanced dataset contains nearly 18,000 images, half of which are vehicles from various angles, and the other half are unrelated images.

Folder in the "./external_data/" directory:
    - "./external_data/vehicles" containing images of vehicles 
    - "./external_data/non-vehicles" containing some negative samples. 


## Submission folder
This folder store the CSV file create by the main_v2.py function and can be directly submitted on Kaggle. 


## Test folder
The folder "./test/test/" contains images to score for submission on Kaggle.


## Train folder 
The folder "./train/" contains a folder "./train/train" containing labelled images. The labels are stored into a file 
"./train/train.csv". One label consists into a sequence of number to group in subsequence of 4 to get (x, y, width, height)
of actual bounding boxes used for training. 


## Data preparation file
This file contains a unique class "ExternalDataPreparation" used to prepare external data for ML models binary classifier. Thus,
calling the function prepare_features of this class directly outputs features and labels (respectively X, y). 
To create features, the method used in this class is the hog features extractor. This choice has been made due to the effectiveness
of this among the others I tried (mostly resizing).
Note that for using this class, you have to specifiy paths to external data folder (vehicles and non-vehicles).


## Sliding window file 
This file contains a unique class "SlidingWinodwDetection". This is one of the main tool of the project. 
Sliding window method is defined by its width, its height and the step size between 2 windows (along x-axis 
and y-axis). 
The main function of this class is the make_magic_works function. Given a model, a way to extract features from an image
the feed the model and an image is able to compute bounding boxes. It outputs a list of tuple of the form (x, y, width, height) 
corresponding to coordinates of bounding boxes. 
The make_magic_works strongly relies on the draw_mask function. This function gets as input the image, and the coordinates of 
windows (from sliding windows) with probability to contains a vehicle above 50% (according to the model) and the corresponding probability. From this information, it draws a mask: each pixel of the mask is first the sum of probabilities of windows 
it belongs to and then the mask is threshold in order to be binarized. At that point, forms obtained by this process are not 
necessarly rectangles. Thus, contours of these forms is extracted. If the form is too small, it is dropped ; else the best rectangle
that can be obtained from the form creates a bounding box. 


### Main_v2 file 
This the main file to run using command: $ python main_v2.py.
The main file contains a class "MyPipeline" that gathers the whole pipeline and a main function to create a CSV file for submission
(automatically stored into submission directory). 
MyPipeline class: the 2 main functions of the pipeline class are "inference" and "plot". Inference function takes as input an image path and return a list of coordinates of bounding boxes. Plot function takes as input an image path and plot an image with bounding 
boxes corresponding to prediction of my whole pipeline. 


### Evaluation file
This file contains a unique class named "PipelineEvaluation". 
"PipelineEvaluation" class design a strategy to evaluate pipeline given an inference function. To create the class you need to precise the path to labeled images (train_path) and the path to the csv file contianing labels/actual boxes (the train.csv file for example).
The metric used to evaluate the pipeline is the Sorensen-Dice coefficient (also used for evaluation on Kaggle). However, as the cost of ocmputation is quite huge of this class as it has for each image to compute bounding boxes and the compute Dice coefficient score, you can limit the nb of images you want to use for scoring. 


### svm_classifier.pkl
This file contains the parameters of a trained svm classifier using external_data images. 