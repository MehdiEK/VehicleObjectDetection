"""
This file contains class and function used to preprocess external
data. In ExternalDataPreparation class, images are transformed into 
hog features. 
Thus, ExternalDataPreparation().prepare_features() is the main 
function to call in order to prepare external data. 

Creation date: XX/XX/2024
Last modification: 22/01/2024
By: Mehdi EL KANSOULI 
"""
import os 
import numpy as np 
import cv2

from skimage.feature import hog


class ExternalDataPreparation():
    """
    Given path to external images, transform them into array of features
    and array of labels (in order to train binary classifier).
    """
    
    def __init__(self, vehiclepath=None, nonvehiclepath=None):
        """
        Takes as input path to folders containing positive and
        negative values.

        :params vehiclepath: str
            Path to images labelled as positive
        :params nonvehiclepath: str
            Path to images labelled as negative
        """
        self.vehiclepath = vehiclepath
        self.nonvehiclepath = nonvehiclepath

    @staticmethod
    def create_hog_features(img):
        """ 
        Create hog feature from an image.

        :params img: cv2.imread object
        :return np.array
        """
        # Resize image
        img = cv2.resize(img,(96,64))

        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate HOG features 
        features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2))
        
        return features
    
    def extract_features_from_path(self, path):
        """
        Given a path (to a folder) extract hog features 
        from all images in the folder. 

        :params path: os.path object
        :return list
        """
        # get images in indicated path
        images = os.listdir(path)

        # list to store prepared data
        prepared_data = []

        for img_name in images:
            # get image
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)

            # create and store hog features
            features = self.create_hog_features(img)
            prepared_data.append(features)

        return prepared_data
    
    def prepare_features(self):
        """
        Create features using positive and negative path. 
        """
        # extract features from vehicles images
        positive_data = self.extract_features_from_path(self.vehiclepath)
        positive_labels = [1] * len(positive_data)

        # extract features from non vehicles images
        negative_data = self.extract_features_from_path(self.nonvehiclepath)
        negative_labels = [0] * len(negative_data)

        # concat both 
        X = np.array(positive_data + negative_data)
        y = np.array(positive_labels + negative_labels)

        return X, y