"""
This file contains the main class used to perform sliding 
window. 

Creation date: XX/XX/2024
Last modification: 22/01/2024
By: Mehdi EL KANSOULI 
"""

import numpy as np 
import cv2
import matplotlib.pyplot as plt 


class SlidingWindowsDetection():
    """
    The sliding window process is defined by three parameters: step size, 
    width of windows and height of windows. 
    """ 
    def __init__(self, step_size, width, height):
        """
        Given step size, width and height of windows, apply 
        sliding windows strategy.
        :params step_size: int 
            Step size between 2 consecutive windows
        :params width: int
            Width of winodws 
        :params height: int 
            Height of windows.        
        """
        self.step_size = step_size
        self.width = width
        self.height = height

    def sliding_window(self, img):
        """
        Iterator to give windows. 
        Note that according y-axis, for loop st rt at 200 and end 50 pixels 
        before end of image. This is to avoid looking for vehicles in the sky 
        or too close from our vehicle. 

        :params img: 
        
        :yield tuple
        """
        # Iterate over the image with a sliding window
        for y in range(200, img.shape[0] - self.height - 50, self.step_size):
            for x in range(0, img.shape[1] - self.width + 1, self.step_size):
                yield (x, y, self.width, self.height)

    @staticmethod
    def draw_mask(img_path, regions):
        """
        Given an image and windows labelled as positive, draw optimal
        rectangles. 

        :params img_path: os.path object (or str)
            Path to studied image
        :params regions: dictionary
            Keys are windows labelled as candidates for containing
            vehicles and values are proba of containing a vehicle
            (output from a model).

        :return list
            List of rectangles containing vehicles.
        """
        # initalize list of output rectangles
        cars = []

        # open image
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape[:2])

        # create a fist max from windows
        for rectangles, proba in regions.items():

            # get coordinates of rectangle
            x, y, w, h = rectangles

            # increase value of pixels in the windows proportionally to proba
            mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w] + proba * 10
        
        # threshold constructed mask and binarize it
        # mask = np.where(mask <= np.max(mask)/2.5, 0, 1)
        mask = np.where(mask <= max(np.max(mask), 8.)/4, 0, 1)
        mask = np.asarray(mask * 255).astype(np.uint8)

        # get contour on mask 
        contours, _ = cv2.findContours(mask,1,2)[-2:]

        # transform contours into rectangles
        for c in contours:
            # if area to small: drop it 
            if cv2.contourArea(c) < 10*10:
                continue
            # create new rectangle and store it
            (x,y,w,h) = cv2.boundingRect(c)
            rect = x, y, w, h
            cars.append(rect)
            
        return cars
    
    @staticmethod
    def check_with_model(sub_image, features_creator, model):
        """
        Given a window, create features and use model to label the window.

        :params sub_image: image
            Window to label
        :params features_creator: function 
            Function used to create feature from image for model training. 
        :params model:
            Trained classifier.  
        
        :return float
            Probability that the image contains a vehicle. 
        """
        features = features_creator(sub_image)  # create features
        features = np.array(features).reshape(1, -1)  # reshape for pred
        return model.predict_proba(features)[0, 1]  # return proba

    def make_magic_works(self, img_path, features_creator, model):
        """
        Given an image, function used in the whole pipelie to extract features
        from images and a trained model, return coordinates of bounding boxes 
        as a lit of tuples (x, y, width, height).

        :params img_path: str
            Path to image to process
        :prams features_creator: function 
            Takes as input an image (or subimage)
        :params model: model 
            Trained binary classifier with fit function 

        :return list of tuples
            Coordinates of bounding boxes.
        """
        # get image
        img = cv2.imread(img_path)

        # initialize dictionary containing positive windows
        cars = {}

        # start windowing image
        for rect in self.sliding_window(img):

            # get coordinates of rectangle
            x, y, w, h = rect

            # get image corresponding to window coordinates
            sub_image = img[y:y+h, x:x+w]

            # get proba of containing a vehicle (being positive)
            prob = self.check_with_model(sub_image, features_creator, model)

            # keep only window with >= 50% chance of being positive
            if prob >= .5:
                cars[rect] = prob

        return self.draw_mask(img_path, regions=cars)

    def plot_labelled_image(self, img_path, features_creator, model):
        """
        Given an image, function used in the whole pipelie to extract features
        from images and a trained model, plot a labelled image with bounding
        boxes. 

        :params img_path: str
            Path to image to process
        :prams features_creator: function 
            Takes as input an image (or subimage)
        :params model: model 
            Trained binary classifier with fit function 

        :return None 
            Plot labelled image
        """
        # get rectangles
        rectangles = self.make_magic_works(img_path, features_creator, model)

        # get image
        img = cv2.imread(img_path)

        # plot image with boxes
        fig, ax = plt.subplots(figsize=(10, 8))
        for rect in rectangles:
            x, y, w, h = rect
            im = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        ax.imshow(im)

        plt.show()