
import cv2
import numpy as np
import keras
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import load_model

def get_image():
    file = raw_input("Write the file name path or press enter to capture an image: \n")
    image_original = cv2.imread(file, 0)
    return (image_original, file)

# function that resizes the image to fit the model
def resize(image):
    image_resized = cv2.resize(255-image, (28,28))
    return image_resized
#funtion that converts the image to 3 channel color
def to3channel(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
# function that converts to grayscale
def togray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
# converto to binary
def binary(image):
    #thresh_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    return binary
#class prediction
def predict(img):
    binary = np.expand_dims(img, 0)
    binary = np.expand_dims(binary, 3)
    classes = loaded_model.predict_classes(binary, batch_size=5)
    return classes

def main():
    global loaded_model
    loaded_model = load_model('../models/model.h5')
    global image_binary
    global image_original
    global file

    print("Assigment 4: CS-512, COMPUTER VISION. \n")
    print("Author: Diego Martin Crespo. \n")
    print("This program clasifies an image of a number into even or odd classes. \n")

    while(True):
        #load image
        image_original, file = get_image()
        if file == '\x1b':
            cv2.destroyAllWindows()
            print("'Esc' key pressed: Program Closed!\n")
            break
        elif file == 'q':
            cv2.destroyAllWindows()
            print("'q' key pressed: Program Closed!\n")
            break
        elif file == '':
            print('No path specified\n')
        else:
            if str(image_original) == 'None':
                print('Wrong path specified')
            elif(len(image_original) > 0):
                #show original image
                cv2.imshow('Original Image', image_original)
                #resize original to 28x28
                image_original = resize(image_original)
                #to binary
                image_binary = binary(image_original)
                #show binary image
                cv2.imshow('Binary Image', image_binary)
                #predict image type
                prediction = predict(image_binary)
                if prediction[0] == 0:
                    print('Clasified as even or class 0\n')
                elif prediction[0] == 1:
                    print('Clasified as odd or class 1\n')
                else:
                    print('Clasified as unknown\n')
                cv2.waitKey(50)

        #else:
         #   print("Press key to enter a new image or 'q' or 'Esc' to quit")


if __name__ == '__main__':
    main()
