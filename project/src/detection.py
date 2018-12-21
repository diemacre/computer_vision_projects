# -*- coding: utf-8 -*-
import cv2
import numpy as np
from classification import training, getLabel
import os
import argparse
import glob

SIGNS = ["ERROR", 
        "STOP",
        "DO NOT TURN LEFT",
        "DO NOT TURN RIGHT",
        "ONE WAY",
        "SPEED LIMIT",
        "TURN RIGHT",
        "TURN LEFT",
        "CROSS INTERCEPTION",
        "ALERT",
        "SCHOOL",
        "RAIL REDUCTION",
        "PRIORITY",
        "NOT PARKING"]

def contrastLimit(image):
    """
    Improves the contrast of the image, to avoid really bright or dark images.
    """
    img_to_eq = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_to_eq)
    channels[0] = cv2.equalizeHist(channels[0])
    img_to_eq = cv2.merge(channels)
    img_to_eq = cv2.cvtColor(img_to_eq, cv2.COLOR_YCrCb2BGR)
    #cv2.imshow("Contrast", img_to_eq)
    #value = cv2.waitKey(-1)
    
    return img_to_eq


def borders(image):
    """
    Find the borders on the image. Use the bilateral filter to reduce noise 
    while keeping edges sharp then apply edge detection with Canny algorithm.
    Finally it scales and gives the absolute values of the images.
    """
    bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    scale_edge = cv2.convertScaleAbs(edge_detected_image)
    #cv2.imshow("Edges", scale_edge)
    #value = cv2.waitKey(-1)
    return scale_edge

def binarization(image):
    """
    It transforms the given image to binary.
    """
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    #cv2.imshow("Binary", thresh)
    #value = cv2.waitKey(-1)
    return thresh

def preprocess_image(image):
    """
    It applies the previous functions to the imaged passed.
    """
    image = contrastLimit(image)
    image = borders(image)
    image = binarization(image)
    return image

def removeSpots(image, threshold):
    """
    Removes small components from the image that their size is larger 
    than a trheshold 
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img = np.zeros((output.shape),dtype = np.uint8)
    
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img[output == i + 1] = 255
    return img

def findContours(image):
    """
    Finds contours of the image.
    """
    _, contours, _ = cv2.findContours(image ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contourSign(contours):
    """
    Finds the contourns of the image that could be a sign, by recognizing 
    triangles and circle-like objects. 
    """    
    coordinates = []
    for cnt in contours:
        approx = cv2.approxPolyDP(
            cnt, 0.07 * cv2.arcLength(cnt, True), True)
        approxi = cv2.approxPolyDP(
                cnt, 0.01*cv2.arcLength(cnt,True), True)
        if (((len(approx) == 3 )  or (len(approxi) == 3 ) or 
             (len(approx)>7) or (len(approxi) > 7))):
            coordinates.append(cnt)
    return coordinates

def cropSingleContour(image, coordinate):
    """
    Crops the image on a square following some coordinates of a contour.
    """
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    return image[top:bottom,left:right]

def findLargest(image, contours, area_threshold):
    """
    Find the largest object (typically a sign) of the image. 
    """
    max_area = 0
    coordinate = None
    sign = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area and area > area_threshold:
            max_area = area
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+2,bottom+2)]
            sign = cropSingleContour(image,coordinate)
    return sign, coordinate

def findLocation(image, model, count):
    """
    Process the image to obtain the main sign and labels it. Marks it with 
    a rectangle and a text that corresponds with the image.
    Creates a file with the located sign. 
    """
    image1 = preprocess_image(image)
    image1 = removeSpots(image1, 300)
    contours = findContours(image1)
    coord = contourSign(contours)
    sign, coordinate = findLargest(image, coord, 100) 
    text = ""
    sign_type = -1

    if sign is not None:
        sign_type = getLabel(model, sign)
        sign_type = sign_type
        text = SIGNS[sign_type]
        cv2.imwrite("../data/temp/"+str(count)+'_'+text.replace(" ", "_")+'.png', sign)
    
    if sign_type > 0:        
        cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2, cv2.LINE_4)
    return coordinate, image, sign_type, text

def clean_images():
    """
    Cleans the located imaged from previous runs of the program.
    """
    file_list = os.listdir("../data/temp/")
    print(file_list)
    for file_name in file_list:
        if '*.png' in file_name:
            os.remove(file_name)

def main(args):
    """
    Main program. Trains model and tries to find a sign on 
    the image an classify it.
    """
    clean_images()
    model = training()
    
    img_dir = args.dir_name # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv2.imread(f1)
        data.append(img)
    i=0
    count = 0

    while i<len(data):
        frame = data[i]
        i=i+1
        frame = cv2.resize(frame, (int(640),int(480)))
        cv2.imshow('Preview', frame)
        value = cv2.waitKey(-1)
        #cv2.destroyWindow('Preview')
        coordinate, image, sign_type, text = findLocation(frame, model, count)
        print(text)
        if coordinate is not None and sign_type ==0:
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 0, 255), 1)
        cv2.imshow('Result', image)
        count = count + 1
        value = cv2.waitKey(-1)
        cv2.destroyWindow('Result')
        if(value & 0xFF == ord('q')):
            break
    return      
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CV Project 'Traffic Sign' Command Line")   
    parser.add_argument(
      '--dir_name',
      default= "../data/Test",
      help= "Directory to the data to be analyzed. (Default: ../data/Test)"
      )
    args = parser.parse_args()
    main(args)