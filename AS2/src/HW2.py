#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @Title:Assigment 2 of CS-512 - Computer Vision
    @author: Diego Martin Crespo
    @Term: Fall 2018
    @Id: A20432558
    """
import numpy as np
import sys
import math
import cv2

#Function that gets the image file or capture it from camera
def get_image():
# 1. read from file or videocapture
    file = input("Write the file name path or press enter to capture an image: \n")
    if len(file) > 1:
        image_original = cv2.imread(file,1)
    else:
        cap = cv2.VideoCapture(0)
        retval,image_original = cap.read()
# 2. Conversion to 3 color channel
    image_original = to3channel(image_original)

# 3. Any size should work. This part of the program rezises the image so it has the proper size for
#    displayong it in a screen
    while image_original.shape[0] > 1200 or image_original.shape[1] > 750:
	    image_original = cv2.resize(image_original,(int(image_original.shape[1]/2), int(image_original.shape[0]/2)))

    cv2.imshow ('image', image_original)
    return (image_original, file)

def to3channel(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

# 4.a function that reloads the original image, pressed "i"
def reloadimage(image):
    if len(image) > 1:
        image_original = cv2.imread(image)
    else:
        cap = cv2.VideoCapture(0)
        retval,image_original = cap.read()
    while image_original.shape[0] > 1200 or image_original.shape[1] > 750:
        image_original = cv2.resize(image_original,(int(image_original.shape[1]/2), int(image_original.shape[0]/2)))
    return image_original

# 4.b function that save current image to file, pressed "w"
def savef(image):
    cv2.imwrite("out.jpg", image)

# 4.c function that converts to gray with cv2, pressed "g"
def togray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# 4.d function that converts to gray with own implementation, pressed "G"
def tograyown(image):
    aux=np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return aux.astype(np.uint8)

# 4.e function that cycles through color channels, pressed "c"
def changecolorCH(image, count):
    if len(image.shape) == 3:
        if count == 1:
            image[:,:,1] = 0
            image[:,:,2] = 0
            count = 1
            print("blue channel showed")
        elif count == 2:
            image[:,:,0] = 0
            image[:,:,2] = 0
            count = 2
            print("green channel showed")
        else:
            image[:,:,0] = 0
            image[:,:,1] = 0
            count = 1
            print("red channel showed")
    else:
        print("Not possible to convert to b, g or r")
    return image

# 4.f function that smooths the image using opencv, pressed "s"
def smooth(self):
    n = self
    image = reloadimage(file)
    image = togray(image)
    if self != 0:
        kernel = np.ones((n,n), np.float32)/(n*n)
        image = cv2.filter2D(image, -1, kernel)
    cv2.imshow('image', image)

# 4.g function that smooths the image using own implementation with finction "conv", pressed "S"
def smoothown(self):
    aux=0
    if self%2 !=1:
        aux=1
    n = self + aux
    image = reloadimage(file)
    image = tograyown(image)
    if self != 0:
        kernel = np.ones((n,n), np.float32)/(n*n)
        image = conv(image, kernel)
    cv2.imshow('image', image)

#function from https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
def convolve(f, g):
    # f is an image and is indexed by (v, w)
    # g is a filter kernel and is indexed by (s, t),
    #   it needs odd dimensions
    # h is the output image and is indexed by (x, y),
    #   it is not cropped
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    # smid and tmid are number of pixels between the center pixel
    # and the edge, ie for a 5x5 filter they will be 2.
    #
    # The output size is calculated by adding smid, tmid to each
    # side of the dimensions of the input image.
    vmax = f.shape[0]
    wmax = f.shape[1]
    smax = g.shape[0]
    tmax = g.shape[1]
    smid = smax // 2
    tmid = tmax // 2
    xmax = vmax + 2 * smid
    ymax = wmax + 2 * tmid
    # Allocate result image.
    h = np.zeros([xmax, ymax], dtype=f.dtype)
    # Do convolution
    for x in range(xmax):
        for y in range(ymax):
            # Calculate pixel value for h at (x,y). Sum one component
            # for each pixel (s, t) of the filter g.
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h

#own convolution function
def conv(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = int(math.floor(kW/2))
    image = cv2.copyMakeBorder(image, pad, pad , pad , pad , cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad : y + pad +1, x - pad : x + pad +1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
    return output.astype(np.uint8)

# 4.h function that downsamples the image by 2 factor with no smoothing, pressed "d"
def downsamples(image):
    image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
    return image

# 4.i function that downsamples the image by 2 factor with smoothing, pressed "D"
def downsamplens(image):
    image = cv2.pyrDown(image)
    return image

# 4.j & 4.k funtion that makes the convolution with a x derivative filter and normalizaed, pressed "x" or "y"
def convdevnorm(image,xory):
    aux=0
    if (xory==1):
        aux=0
    else:
        aux=1
    sobelx = cv2.Sobel(image,cv2.CV_64F,xory,aux,ksize=5)
    image = cv2.normalize(sobelx, image, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return image

# 4.l function that shows magnitude of the gradient computed with x and y derivatives in range [0,255], pressed "m"
def gradmag(image):
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    gradient = cv2.magnitude(sobelx, sobely)
    image = cv2.normalize(gradient, image, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return image

# 4.m funtion that plots the gradient vectors of the image, pressed "p"
def gradvectors(self):
    n = self
    image = reloadimage(file)
    image = togray(image)
    grad = image
    if self != 0:
        sobelx = cv2.Sobel(grad,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(grad,cv2.CV_64F,0,1,ksize=5)
        for x in range(0, grad.shape[0], n):
            for y in range(0, grad.shape[1], n):
                grad_angle = math.atan2(sobely[x, y], sobelx[x, y])
                grad_x = int(x + n * math.cos(grad_angle))
                grad_y = int(y + n * math.sin(grad_angle))
                cv2.arrowedLine(grad, (y, x), (grad_y, grad_x), (0, 0, 0))
    cv2.imshow('image', grad)
    image = grad

# 4.n funtion that rotates the image, pressed "r"
def rotate(self):
    n = self
    image = reloadimage(file)
    image = togray(image)
    rotated = image
    if self != 0:
        rot = cv2.getRotationMatrix2D((rotated.shape[1]/2, rotated.shape[0]/2), n, 1)
        rotated = cv2.warpAffine(rotated, rot,(rotated.shape[1], rotated.shape[0]))
    cv2.imshow('image', rotated)
    image = rotated

# 4.o function that prints help about the use of the keyboard keys for the program , pressed "h"
def help():
    print("Press ‘i’ to reload the original image. \n")
    print("Press ‘w’ to save the current image into the file 'ouput.jpg' \n")
    print("Press ‘g’ to convert the image to grayscale using the OpenCV conversion function \n")
    print("Press ‘G’ to convert the image to grayscale using your implementation of conversion function. \n")
    print("Press ‘c’ to cycle through the color channels of the image showing a different channel every time the key is pressed. \n")
    print("Press ‘s’ to convert the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing. \n")
    print("Press ‘S’ to convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter. Use a track bar to control the amount of smoothing, \n")
    print("Press ‘d’ to downsample the image by a factor of 2 without smooting. \n")
    print("Press ‘D’ to downsample the image by a factor of 2 with smoothing. \n")
    print("Press ‘x’ to convert the image grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255]. \n")
    print("Press ‘y’ to convert the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255]. \n")
    print("Press ‘m’ to show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed base don the x and y derivatives of the image. \n")
    print("Press ‘p’ to convert the image to grayscale and plot the gradient vectors of the image every N pixel and let the plotted gradient vectors have a lenght of K. Use a track bar to control N. Plot the vectors as short line segments of length K. \n")
    print("Press ‘r’ to convert the image to grayscale and rotate it using an angle of teta degrees. Use a track bar to control the rotation angle. \n")
    print("Press ‘h’ to display a short description of the program, its command line arguments, and the keys it supports. \n")
        
def main():
    global image
    global file
    image, file = get_image()
    print("Assigment 2: CS-512, COMPUTER VISION. \n")
    print("Author: Diego Martin Crespo. \n")
    print("This program ables to manipulate images mainly using the openCV library. \n")
    print("Press h for help")
    count = 0
    while(True):
        key = cv2.waitKey()
        print (key)
        if key == ord('i'):
            image = reloadimage(file)
            print("'i' key pressed: image reloaded")
        
        elif key == ord('w'):
            savef(image)
            print("'w' key pressed: image saved into out.jpg file")
        
        elif key == ord('g'):
            image = reloadimage(file)
            image = togray(image)
            print("'g' key pressed: convert to grey using openCV method")  
        
        elif key == ord('G'):
            image = reloadimage(file)
            image = tograyown(image)
            print("'G' key pressed: convert to grey using own method")
        
        elif key == ord('c'):
            count= count + 1
            if count > 2:
                count = 0
            image = reloadimage(file)
            print("'c' key pressed:")
            image = changecolorCH(image, count)

        elif key == ord('s'):
            image = reloadimage(file)
            image = togray(image)
            cv2.imshow('image', image)
            print("'s' key pressed: Image to grayscale and smooth track bar created using openCV")
            cv2.createTrackbar('s', 'image', 0, 255, smooth)
        
        elif key == ord('S'):
            image = reloadimage(file)
            image = tograyown(image)
            cv2.imshow('image', image)
            print("'S' key pressed: Image to grayscale and smooth track bar created using own method")
            cv2.createTrackbar('S', 'image', 0, 20, smoothown)
        
        elif key == ord('d'):
            image = reloadimage(file)
            image = downsamples(image)
            print("'d' key pressed: downsample 2 factor with no smoothing")
        
        elif key == ord('D'):
            image = reloadimage(file)
            image = downsamplens(image)
            print("'D' key pressed: downsample 2 factor with smoothing")
        
        elif key == ord('x'):
            image = reloadimage(file)
            image = togray(image)
            image = convdevnorm(image, 1)
            print("'x' key pressed: convert to grayscale, convolution with x derivative filter")

        elif key == ord('y'):
            image = reloadimage(file)
            image = togray(image)
            image = convdevnorm(image, 0)
            print("'y' key pressed: convert to grayscale, convolution with x derivative filter")

        elif key == ord('m'):
            image = reloadimage(file)
            image = togray(image)
            image = gradmag(image)
            print("'m' key pressed: show magnitude of gradient computed with x and y derivatives to range [0,255]")
        
        elif key == ord('p'):
            image = reloadimage(file)
            image = togray(image)
            print("'p' key pressed: convert image to grayscale and plot gradient vectors")
            cv2.createTrackbar('p', 'image', 0,255, gradvectors)
        
        elif key == ord('r'):
            image = reloadimage(file)
            image = togray(image)
            print("'r' key pressed: convert image to grayscale and rotate x degrees")
            cv2.createTrackbar('r', 'image', 0, 360, rotate)
        
        elif key == ord('h'):
            print("'h' key pressed: description of key functions:")
            help()

        elif key == 27:
            cv2.destroyAllWindows()
            print("'Esc' key pressed: Program Closed!")
            break
        else:
            print("Wrong key pressed, press 'h' for help")
        cv2.imshow('image', image)



if __name__ == '__main__':
    main()
