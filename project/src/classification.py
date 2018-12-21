import cv2
import numpy as np
from os import listdir
#Parameters
SIZE = 32
CLASS_NUMBER = 14
MAX_FILES = 80


def load_traffic_dataset():
    """
    Function that loads the data from the source folder for training the model
    """
    dataset = []
    labels = []
    for sign_type in range(CLASS_NUMBER):
        #if(sign_type != 0):
        sign_list = listdir("../data/Samples/{}".format(sign_type))
        i=0
        for sign_file in sign_list:
            if(i<MAX_FILES):
                if ('.png') in sign_file:
                    path = "../data/Samples/{}/{}".format(sign_type,sign_file)
                    #print(path)
                    img = cv2.imread(path,0)
                    img = cv2.resize(img, (SIZE, SIZE))
                    img = np.reshape(img, [SIZE, SIZE])
                    dataset.append(img)
                    labels.append(sign_type)
                    i+=1
            else: 
                break
    return np.array(dataset), np.array(labels)

def get_hog():
    """
    HogDecriptor function implemented using the OpenCV function
    """ 
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 5
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                            derivAperture,winSigma,histogramNormType,
                            L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog

class SVM():
    """
    SVM class with functions for the model
    """
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

    def save(self, fn):
        self.model.save(fn)


def deskew(img):
    """
    Function that deskews the images 
    """
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SIZE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def training():
    """
    Function that trains the SVM model using 90% of the samples. It saves the model as a .dat file.
    """
    print('Loading data from data.png ... ')
    # Load data.
    #data, labels = load_data('data.png')
    data, labels = load_traffic_dataset()
    print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(14)
    shuffle = rand.permutation(len(data))
    data, labels = data[shuffle], labels[shuffle]
    # Deskwew images for fixing possible issues on the images
    print('Deskew images ... ')
    data_deskewed = list(map(deskew, data))

    # HoG feature descriptor
    print('Defining HoG parameters ...')
    hog = get_hog()
    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    for img in data_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    
    print('Spliting data into training (90%) and test set (10%)... ')
    train_n=int(0.9*len(hog_descriptors))
    data_train, data_test = np.split(data_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])
    
    print('Training SVM model ...')
    model = SVM()
    model.train(hog_descriptors_train, labels_train)
    hog_descriptors_test=np.reshape(hog_descriptors_test, [-1, hog_descriptors_test.shape[1]])
    evaluate_model(model, data_test, hog_descriptors_test, labels_test)

    print('Saving SVM model ...')
    model.save('../models/data_svm.dat')
    return model

def getLabel(model, data):
    """
    Funtion that gets the label of the image by prediction using the trained model
    """
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = [cv2.resize(gray,(SIZE,SIZE))]
    img_deskewed = list(map(deskew, img))
    hog = get_hog()
    hog_descriptors = np.array([hog.compute(img_deskewed[0])])
    hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
    return int(model.predict(hog_descriptors)[0])

def evaluate_model(model, data, samples, labels):
    resp = model.predict(samples)
    print(resp)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((14, 14), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    #print('confusion matrix:')
    #print(confusion)

def get_image():
    """
    Funtion that test an image given as an input in the terminal. Not used in the main program (detection.py)
    """
    file = input("Write the file name path or press enter to capture an image: \n")
    image_original = cv2.imread(file, 1)
    return (image_original, file)

def main():
    """
    Main funtion for testing the classification.py program. Not used in the main program (detection.py)
    """
    image = get_image()[0]
    model = training()
    result= getLabel(0, image)
    print(result)

if __name__ == '__main__':
    main()

