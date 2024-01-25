"""
This is the main file to run in order to submit for kaggle competition. 
This file construcut the pipeline through the class MyPipeline and also 
submission csv file using main function defined below. 

Command to run: $ python main_v2.py

Creation date: XX/XX/2024
Last modification: 22/01/2024
By: Mehdi EL KANSOULI 
"""
import os 
import pickle
import pandas as pd 

from sklearn.svm import SVC

from utils import * 
from data_preparation import ExternalDataPreparation
from sliding_window import SlidingWindowsDetection


class MyPipeline(SlidingWindowsDetection):
    """
    End-to-end pipeline used for vehicle detection. 
    In particular, this class contains an inference function wich is also 
    the main function of the project taking as input an image and returning 
    coordinates of bounding boxes. 
    """

    def __init__(self, features_extractor, model, step_size=50, 
                 width=200, height=150):
        """
        Initialization of the class.

        :params features extractor: function
            Given an image, create features
        :params model: model
            Trained classifier
        :params step_size: int 
            Step size for sliding window
        :params width: int
            Wisth of windows
        :params height: int 
            Height of window 
        """
        super().__init__(step_size=step_size, width=width, height=height)
        self.make_features = features_extractor
        self.model = model

    def inference(self, img_path):
        """
        Run inference. 
        Given an image, return bounding boxes.

        :params img_path: str
            Path to image
        
        :return list of tuple
        """
        return self.make_magic_works(img_path, 
                                     features_creator=self.make_features,
                                     model=self.model)
    
    def plot(self, img_path):
        """
        Given an image, plot image with bounding boxes. 

        :params img_path: str
            Path to image. 
        """
        self.plot_labelled_image(img_path, 
                                 features_creator=self.make_features, 
                                 model=self.model)
        

def main(external_data="./external/", saved_model="./svm_classifier.pkl",
         step_size=50, width=200, height=150, test_dir='./test/test/'):
    """
    Main function to run in order to get a submission file. 

    :params external_data: str, default='./external/"
        Path to external data
    :params saved_model: str, default="./svm_classifier.pkl"
        Path to saved model, if None retrain an SVM classifier.
    :params step_size: int, default=50
        Step size for sliding window
    :params width: int, default=200
        Width of windows
    :params height: int, default=150
        Height of winodws
    :params test_dir: str, default="./test/test/"
        Path to folder containing files to score. 

    :return None
        Create a CSV file
    """
    # data preparation 
    vehiclepath = os.path.join(external_data, "vehicles")
    nonvehiclepath = os.path.join(external_data, "non-vehicles")
    preparator = ExternalDataPreparation(vehiclepath, nonvehiclepath)
    
    # open or create classifier
    if saved_model is None:
        my_model = SVC(probability=True)
        X, y = preparator.prepare_features()
        my_model.fit(X, y)
    else:
        with open(saved_model, 'rb') as file:
            my_model = pickle.load(file)

    # define pipeline 
    pipeline = MyPipeline(
        features_extractor=preparator.create_hog_features, 
        model=my_model, 
        step_size=step_size, 
        width=width, 
        height=height
    )

    # create submission file
    rows = []  # initialize submission 
    test_files = sorted(os.listdir(test_dir))
    nb_files = len(test_files)
    i=1

    for file_name in test_files:

        # get image path
        file_path = os.path.join(test_dir, file_name)
        
        # get bounding boxes
        candidates = pipeline.inference(file_path)
        
        # encode the bounding boxes detected for the frame
        mask = bounding_boxes_to_mask(candidates, H, W)
        rle = run_length_encoding(mask)

        #append the predicted bounding boxes to your results' list
        rows.append(['test/' + file_name, rle])
        
        # print how much images are ready 
        if i % 10 == 0 or i == nb_files:
            print(f"{i} / {nb_files}: Done")
        i += 1
    
    # generate csv submission
    df_prediction = pd.DataFrame(columns=['Id', 'Predicted'], data=rows).set_index('Id')
    df_prediction.to_csv('./submission/sample_submission.csv')


if __name__ == '__main__':
    main(
        step_size = 18,
        width = 64*3,
        height = 36*2, 
    )
    

