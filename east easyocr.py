import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import cv2
import easyocr
import os
import time


# Replace the argparse section with the following lines
image_folder_path =  r'C:\Users\sreek\Desktop\Mini project 2\OCR images' # Provide the path to your image folder
east_path = r'C:\Users\sreek\Desktop\Mini project 2\EAST\model.py'# Provide the path to your EAST model

min_confidence = 0.5
width = 320
height = 320

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Loop through each image in the folder
for image_name in os.listdir(image_folder_path):
    if image_name.endswith('.jfif'):
        # Load the input image and grab the image dimensions
        image_path = os.path.join(image_folder_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load the image from {image_path}")
            continue

        orig = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (width, height)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(east_path)

        # construct a blob from the image and then perform a forward pass
        # of the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)

        # Measure the time taken for text detection
        start = time.time()
        (scores, geometry) = net.forward(layerNames)
        end = time.time()
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        # ... (rest of the script remains unchanged)

        # Use EasyOCR to perform OCR on the image
        results = reader.readtext(orig)

        # Display the detected text
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            cv2.putText(orig, text, (int(top_left[0]), int(top_left[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show the output image
        cv2_imshow(orig)
