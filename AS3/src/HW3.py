#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @Title:Assigment 3 of CS-512 - Computer Vision
    @author: Diego Martin Crespo
    @Term: Fall 2018
    @Id: A20432558
    """
import numpy as np
import sys
import math
import cv2
import scipy.stats as st
import os

#Function that gets the image file or capture it from camera
def get_image():
    # read from file or videocapture image1
    file1 = input(
        "Write the file name path or press enter to capture an image1: \n")
    if len(file1) > 1:
        image1 = cv2.imread(file1, 0)
    else:
        cap1 = cv2.VideoCapture(0)
        image1 = cap1.read()[1]
    # Conversion to 3 color channel
    image1 = to3channel(image1)
    # Any size should work. This part of the program rezises the image1 so it has the proper size for
    # displayong it in a screen
    while image1.shape[0] > 1200 or image1.shape[1] > 750:
	    image1 = cv2.resize(image1, (int(
	        image1.shape[1]/2), int(image1.shape[0]/2)))
    # read from file or videocapture image2
    file2 = input(
        "Write the file name path or press enter to capture an image2: \n")
    if len(file2) > 1:
        image2 = cv2.imread(file2, 0)
    else:
        cap2 = cv2.VideoCapture(0)
        image2 = cap2.read()[1]
    # Conversion to 3 color channel
    image2 = to3channel(image2)
    # Any size should work. This part of the program rezises the image2 so it has the proper size for
    # displayong it in a screen
    while image2.shape[0] > 1200 or image2.shape[1] > 750:
            image2 = cv2.resize(image2, (int(
                image2.shape[1]/2), int(image2.shape[0]/2)))
    return (image1, file1, image2, file2)
#funtion that converts to 3 channels
def to3channel(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
# function that reloads the original image, pressed "i"
def reloadimage(file1, file2):
    if len(file1) > 1:
        image1 = cv2.imread(file1,0)
    else:
        cap1 = cv2.VideoCapture(0)
        image1 = cap1.read()[1]
    image1 = to3channel(image1)
    while image1.shape[0] > 1200 or image1.shape[1] > 750:
        image1 = cv2.resize(image1, (int(
            image1.shape[1]/2), int(image1.shape[0]/2)))
    
    if len(file2) > 1:
        image2 = cv2.imread(file2,0)
    else:
        cap2 = cv2.VideoCapture(0)
        image2 = cap2.read()[1]
    image2 = to3channel(image2)
    while image2.shape[0] > 1200 or image2.shape[1] > 750:
        image2 = cv2.resize(image2, (int(
            image2.shape[1]/2), int(image2.shape[0]/2)))
    return (image1, image2)
# function that save current image to file, pressed "w"
def savef(image1, image2):
    cv2.imwrite("out1.jpg", image1)
    cv2.imwrite("out2.jpg", image2)
# function that converts to gray with cv2, pressed "g"
def togray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
#########################################SLIDERS#########################################################
#sliders for harris()- own implementation of corner detection
def k(self):
    if self > 0 and self < 10:
        ownHarris(file1, file2)
    return
def threshold(self):
    if self != 0 and self < 10:
        ownHarris(file1, file2)
    return
def sizewind(self):
    if self != 0:
        ownHarris(file1, file2)
    return
####################################################
#sliders for cornerharri s- opencv implementation of corner detection
def k2(self):
    if self > 0 and self < 10:
        cornerHarris(file1, file2)
    return
def threshold2(self):
    if self != 0 and self < 10:
        cornerHarris(file1, file2)
    return
def sizewind2(self):
    if self != 0:
        cornerHarris(file1, file2)
    return
####################################################
#sliders for features
def k3(self):
    if self > 0 and self < 10:
        features(file1, file2)
    return
def threshold3(self):
    if self != 0 and self < 10:
        features(file1, file2)
    return
def sizewind3(self):
    if self != 0:
        features(file1, file2)
    return
####################################FUNCTIONS-CORNER-DETECTION############################################
#own implementation of corner detection function, press 'x'
def ownHarris(file1, file2):
    img1 = reloadimage(file1, file2)[0]

    k = 2*cv2.getTrackbarPos('k', 'Image_own_processing')/1000
    threshold = cv2.getTrackbarPos('Threshold', 'Image_own_processing')/10
    winsize = cv2.getTrackbarPos('Window_Size', 'Image_own_processing')
    
    max_r = 0
    r = []
    M = np.matrix([[], []])
    corner_list = []

    img1sx = img1
    img1sobelx = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    img1sx = cv2.normalize(img1sobelx, img1sx, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    img1sx = img1sx**2
    img1sy = img1
    img1sobely = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
    img1sy = cv2.normalize(img1sobely, img1sy, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    img1sy = img1sy**2
    img1sxy = img1sx*img1sy

    for i in range(math.floor(winsize/2), (img1sx.shape[0])-math.floor(winsize/2)):
        for j in range(math.floor(winsize/2), (img1sx.shape[1])-math.floor(winsize/2)):
            Ix2 = img1sx[i-math.floor(winsize/2):i+math.floor(winsize/2)+1,
                         j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            Iy2 = img1sy[i-math.floor(winsize/2):i+math.floor(winsize/2)+1,
                         j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            IxIy = img1sxy[i-math.floor(winsize/2):i+math.floor(
                winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]

            Sx = Ix2.sum()
            Sy = Iy2.sum()
            Sxy = IxIy.sum()

            M = np.matrix([[Sx, Sxy], [Sxy, Sy]])
            det = np.linalg.det(M)
            tr = np.trace(M)
            r.append([i, j, det - k*tr**2])

    for pixel in r:
        if pixel[2] > max_r:
            max_r = pixel[2]

    for pixel in r:
        if pixel[2] > threshold*max_r:
            corner_list.append((pixel[1], pixel[0]))

    img1 = to3channel(img1)

    while corner_list:
        corner = corner_list.pop()
        cv2.rectangle(img1, (corner[0]-2, corner[1]+2),
                      (corner[0]+2, corner[1]-2), (0, 0, 255), 1)

    cv2.imshow('Image_own_processing', img1)
    print("k_own= ", k)
    print("Threshold_own = ", threshold)
    print("Window_Size_own = ", winsize)

    return
#openCV corner detection function, press 'X'
def cornerHarris(file1, file2):
    img1 = reloadimage(file1, file2)[1]
    gray = togray(img1)
    img1 = togray(img1)
    img1 = to3channel(img1)
    gray = np.float32(gray)

    ksize = cv2.getTrackbarPos('k', 'OpenCV_processing_image')
    blockSize = cv2.getTrackbarPos('Window_Size', 'OpenCV_processing_image')
    threshold = cv2.getTrackbarPos('Threshold', 'OpenCV_processing_image')/1000
    aux = 0
    if ksize % 2 == 0:
        aux = 1
    ksize = ksize + aux
    aux = 0
    if blockSize % 2 == 0:
        aux = 1
    blockSize = blockSize + aux
    cv2.imshow('OpenCV_processing_image', img1)
    # cornerHarris(image, blocksize, ksize, k)
    # blockSize – size of the windows considered for the corner detection-2
    # ksize – parameter for the derivative of Sobel-3
    # k – free parameter for the Harris equation.
    dst = cv2.cornerHarris(gray, blockSize, ksize, 0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image - 0.001
    img1[dst > threshold*dst.max()] = [0, 0, 255]

    print("k OpenCV= ", ksize)
    print("Threshold OpenCV= ", threshold)
    print("Window_Size OpenCV= ", blockSize)
    cv2.imshow('OpenCV_processing_image', img1)
#features using own implementation, press 'f'
def features(file1, file2):
    image1, image2 = reloadimage(file1,file2)

    image1_g = togray(image1)
    image2_g = togray(image2)

    k = cv2.getTrackbarPos('k', 'Features')/1000
    threshold = cv2.getTrackbarPos('Threshold', 'Features')/10
    winsize = cv2.getTrackbarPos('Window_Size', 'Features')

    max_r1 = 0
    r1 = []
    M1 = np.matrix([[], []])
    corner_list1 = []

    max_r2 = 0
    r2 = []
    M2 = np.matrix([[], []])
    corner_list2 = []

    #calculate conerners for image1
    image1sx = image1_g
    image1sobelx = cv2.Sobel(image1_g, cv2.CV_64F, 1, 0, ksize=5)
    image1sx = cv2.normalize(image1sobelx, image1sx, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image1sx = image1sx**2
    
    image1sy = image1_g
    image1sobely = cv2.Sobel(image1_g, cv2.CV_64F, 0, 1, ksize=5)
    image1sy = cv2.normalize(image1sobely, image1sy, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image1sy = image1sy**2

    image1sxy = image1sx*image1sy

    for i in range(math.floor(winsize/2), (image1sx.shape[0])-math.floor(winsize/2)):
        for j in range(math.floor(winsize/2), (image1sx.shape[1])-math.floor(winsize/2)):
            Ix2_1 = image1sx[i-math.floor(winsize/2):i+math.floor(
                winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            Iy2_1 = image1sy[i-math.floor(winsize/2):i+math.floor(
                winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            IxIy_1 = image1sxy[i-math.floor(winsize/2):i+math.floor(
                winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]

            Sx_1 = Ix2_1.sum()
            Sy_1 = Iy2_1.sum()
            Sxy_1 = IxIy_1.sum()

            M1 = np.matrix([[Sx_1, Sxy_1], [Sxy_1, Sy_1]])
            det1 = np.linalg.det(M1)
            tr1 = np.trace(M1)
            r1.append([i, j, det1 - k*tr1**2])

    for pixel in r1:
        if pixel[2] > max_r1:
            max_r1 = pixel[2]

    for pixel in r1:
        if pixel[2] > threshold*max_r1:
            corner_list1.append((pixel[1], pixel[0]))

    #calculate conerners for image2
    image2sx = image2_g
    image2sobelx = cv2.Sobel(image2_g, cv2.CV_64F, 1, 0, ksize=5)
    image2sx = cv2.normalize(image2sobelx, image2sx, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image2sx = image2sx**2

    image2sy = image2_g
    image2sobely = cv2.Sobel(image2_g, cv2.CV_64F, 0, 1, ksize=5)
    image2sy = cv2.normalize(image2sobely, image2sy, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    image2sy = image2sy**2

    image2sxy = image2sx*image2sy

    for i in range(math.floor(winsize/2), (image2sx.shape[0])-math.floor(winsize/2)):
        for j in range(math.floor(winsize/2), (image2sx.shape[1])-math.floor(winsize/2)):
            Ix2_2 = image2sx[i-math.floor(winsize/2):i+math.floor(
                winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            Iy2_2 = image2sy[i-math.floor(winsize/2):i+math.floor(
                winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]
            IxIy_2 = image2sxy[i-math.floor(winsize/2):i+math.floor(
                winsize/2)+1, j-math.floor(winsize/2):j+math.floor(winsize/2)+1]

            Sx_2 = Ix2_2.sum()
            Sy_2 = Iy2_2.sum()
            Sxy_2 = IxIy_2.sum()

            M2 = np.matrix([[Sx_2, Sxy_2], [Sxy_2, Sy_2]])
            det2 = np.linalg.det(M2)
            tr2 = np.trace(M2)
            r2.append([i, j, det2 - k*tr2**2])

    for pixel in r2:
        if pixel[2] > max_r2:
            max_r2 = pixel[2]

    for pixel in r2:
        if pixel[2] > threshold*max_r2:
            corner_list2.append((pixel[1], pixel[0]))

    #calculate features
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    image3 = cv2.drawMatches(image1, kp1, image2, kp2,
                            matches[:20], None, flags=2)
    cv2.imshow('Features', image3)

    #draw corners
    while corner_list1:
        corner1 = corner_list1.pop()
        cv2.rectangle(image3, (corner1[0]-2, corner1[1]+2),
                      (corner1[0]+2, corner1[1]-2), (0, 0, 255), 1)

    while corner_list2:
        corner2 = corner_list2.pop()
        cv2.rectangle(image3, (image1.shape[1]+corner2[0]-2, corner2[1]+2),
                      (image1.shape[1]+corner2[0]+2, corner2[1]-2), (0, 0, 255), 1)

    cv2.imshow('Features', image3)
    print("k = ", k)
    print("Threshold = ", threshold)
    print("Window_Size = ", winsize)

    return
##########################################################################################################
#help funtion for print keys functions
def help():
    print("Press ‘i’ to reload the original image. \n")
    print("Press ‘s’ to save the current image into the file 'ouput.jpg' \n")
    print("Press 'X' to detect corners using OpenCV funtionskey pressed")
    print("Press 'x' to detect corners using own implementation")
    print("Press 'f' to display features image")
    print("Press ‘h’ to display a short description of the program, its command line arguments, and the keys it supports. \n")
#main 
def main():
    global image1
    global image2
    global file1
    global file2
    global image3
    image1, file1, image2, file2 = get_image()
    image3 = np.concatenate((image1, image2), axis=1)

    cv2.imshow('Image_own_processing', image1)
    cv2.imshow('OpenCV_processing_image', image2)
    cv2.imshow('Features', image3)

    print("Assigment 3: CS-512, COMPUTER VISION. \n")
    print("Author: Diego Martin Crespo. \n")
    print("This program is used for corned detection on images. \n")
    print("Press h for help")

    while(True):
        key = cv2.waitKey()
        print(key)

        if key == ord('i'):
            image1, image2 = reloadimage(file1, file2)
            print("'i' key pressed: image reloaded")

        elif key == ord('w'):
            savef(image1, image2)
            print("'w' key pressed: image saved into out.jpg file")

        elif key == ord('x'):
            print("'x' key pressed: detection of corners using own implementation function:")
            cv2.createTrackbar('k', 'Image_own_processing', 20, 80, k)
            cv2.createTrackbar('Threshold', 'Image_own_processing', 0, 10, threshold)
            cv2.createTrackbar('Window_Size', 'Image_own_processing', 0, 10, sizewind)

        elif key == ord('X'):
            print("'X' key pressed: detection of corners using OpenCV:")
            cv2.createTrackbar('k', 'OpenCV_processing_image', 1, 20, k2)
            cv2.createTrackbar('Threshold', 'OpenCV_processing_image',1, 20, threshold2)
            cv2.createTrackbar('Window_Size', 'OpenCV_processing_image', 1, 20, sizewind2)

        elif key == ord('f'):
            print("'f' key pressed: used features function:")
            cv2.createTrackbar('k', 'Features', 20, 80, k3)
            cv2.createTrackbar('Threshold', 'Features', 0, 10, threshold3)
            cv2.createTrackbar('Window_Size', 'Features', 0, 10, sizewind3)

        elif key == ord('h'):
            print("'h' key pressed: description of key functions:")
            help()
        elif key == 27:
            cv2.destroyAllWindows()
            print("'Esc' key pressed: Program Closed!")
            break
        else:
            print("Wrong key pressed, press 'h' for help")

if __name__ == '__main__':
    main()
