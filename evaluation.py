"""
This file contains the whole code used to evaluate my final 
pipeline inference function wich takes as input an image path
and return coordinates of bounding boxes. 
The evaluation is done using the Sorensen-Dice coefficient 
which is also used for Kaggle ranking.

Creation date: XX/XX/2024
Last modification: 22/01/2024
By: Mehdi EL KANSOULI
"""
import os 
import numpy as np 
import pandas as pd 


class PipelineEvaluation():
    """
    This class aims to provide a framework to test the whole pipeline easily.
    In particular, it contains functions used to compute SOrensen-Dice 
    coefficient. This score will also be used to rank the competition. 
    """
    
    def __init__(self, train_path, label_path):
        """
        Initialize the PipelineEvaluation class. 

        :params train_path: str
            Path to the folder containing images 
        :params label_path: str
            Path to file containing coordinates of actual bounding 
            boxes.
        """
        self.train_path = train_path
        self.train_files = os.listdir(train_path)
        self.df_label = pd.read_csv(label_path)
        
        # update dataframe of labels
        correct_path = lambda x: os.path.join(train_path, 
                                              os.path.basename(x))
        self.df_label['frame_id'] = self.df_label.frame_id.apply(correct_path)
        
    @staticmethod
    def annotations_for_frame(df_annotation, frame):
        """
        Function from tools provided on Kaggle. 
        """
        bbs = df_annotation[df_annotation.frame_id == frame].bounding_boxes.values[0]
        bbs = str(bbs).split(' ')
        if len(bbs)<4:
            return []

        bbs = list(map(lambda x : int(x),bbs))

        return np.array_split(bbs, len(bbs) / 4)

    def average_dice_coeff_v2(self, ground_truth_boxes, predicted_boxes):
        """
        Calculate the average Sorensen-Dice coefficient for multiple bounding boxes.

        :params ground_truth_boxes: list of tuples
            Coordinates of the first bounding box.
        :params predicted_boxes: list of tuples
            Coordinates of the second bounding box.

        :return float
            Average Sorensen-Dice coefficient.
        """
        def calculate_intersection_area(bbox1, bbox2):
            # calculate intersection area between two boxes.
            x_left = max(bbox1[0], bbox2[0])
            y_top = max(bbox1[1], bbox2[1])
            x_right = min(bbox1[2]+bbox1[0], bbox2[2]+bbox2[0])
            y_bottom = min(bbox1[3]+bbox1[1], bbox2[3]+bbox2[1])

            if x_right < x_left or y_bottom < y_top:
                return 0  # No overlap

            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            return intersection_area

        def calculate_bbox_area(bbox):
            # calculate area of one box
            return bbox[3] * bbox[2]

        total_intersection_area = 0
        total_bbox_area = 0

        # compute dice coeff
        for bbox1 in predicted_boxes:
            total_bbox_area += calculate_bbox_area(bbox1)
            for bbox2 in ground_truth_boxes:
                total_intersection_area += calculate_intersection_area(bbox1, bbox2)

        if total_intersection_area == 0:
            dice_coefficient = 0
        else:
            dice_coefficient = (2 * total_intersection_area) / (total_bbox_area + sum(map(calculate_bbox_area, ground_truth_boxes)))

        return dice_coefficient

    def compute_metric(self, inference, limit=1000):
        """
        Given an inference function that takes as input an image and 
        outputs bounding boxes, return Sorensen-Dice coefficient. 

        :params inference: function 
            Takes as input an img and return bounding boxes
        :params limit: int, default=1000
            Maximum nb of images used for computing score.

        :return float
        """
        coeff = []  # list for storing score on each image.

        # sample random file to compute score
        nb = np.random.randint(0, len(self.train_files), size=limit)

        for n in nb:
            # select file 
            file = self.train_files[n]

            # get path to file
            file_path = os.path.join(self.train_path, file)
            
            # compute bouding boxes candidates
            candidates = inference(file_path)

            # get actual bouding boxes.
            actual_boxes = self.annotations_for_frame(self.df_label, file_path)

            # compute dice coeff
            dice = self.average_dice_coeff_v2(candidates, actual_boxes)
            coeff.append(dice)

        return np.mean(coeff)