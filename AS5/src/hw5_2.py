import sys
import cv2
import numpy as np
import scipy.stats as st
import os
import math
import glob
import random
import pandas as pd

def calculateV(points3D,points2D):
    
    A = np.zeros((len(points3D)*2, 9))
    j = 0
    for i in range(0, len(points3D)):
        x = np.array([points3D[i][0], points3D[i][1], 1, 0, 0, 0, -points2D[i][0]*points3D[i][0], -points2D[i][0]*points3D[i][1], -points2D[i][0]*1])
        y = np.array([0, 0, 0, points3D[i][0], points3D[i][1], 1, -points2D[i][1]*points3D[i][0], -points2D[i][1]*points3D[i][1], -points2D[i][1]*1])
        A[j] = x
        A[j+1] = y
        j+= 2

    U1, V1, H1 = np.linalg.svd(A, full_matrices=True)
    
    H = H1[8]
    #print(H)

    V12 = [H[0]*H[1], H[0]*H[4]+H[3]*H[1],
           H[3]*H[4], H[6]*H[1]+H[0]*H[7],
           H[6]*H[4]+H[3]*H[7], H[6]*H[7]]

    V11 = [H[0]*H[0], H[0]*H[3]+H[3]*H[0],
           H[3]*H[3], H[6]*H[0]+H[0]*H[6],
           H[6]*H[3]+H[3]*H[6], H[6]*H[6]]

    V22 = [H[1]*H[1], H[1]*H[4]+H[4]*H[1],
           H[4]*H[4], H[7]*H[1]+H[1]*H[7],
           H[7]*H[4]+H[4]*H[7], H[7]*H[7]]
    return V12, V11, V22, H

def calculaparameters(points3D0, points2D0, points3D1, points2D1, points3D2, points2D2):

    A = np.zeros((len(points3D0)*2, 9))
    j = 0
    for i in range(len(points3D0)):
        x = np.array([points3D0[i][0], points3D0[i][1], 1, 0, 0, 0, -points2D0[i][0]*points3D0[i][0], -points2D0[i][0]*points3D0[i][1], -points2D0[i][0]*1])
        y = np.array([0, 0, 0, points3D0[i][0], points3D0[i][1], 1, -points2D0[i][1]*points3D0[i][0], -points2D0[i][1]*points3D0[i][1], -points2D0[i][1]*1])
        A[j] = x
        A[j+1] = y
        j+= 2
    #print(A[0])

    U1, V1, H1 = np.linalg.svd(A, full_matrices=True)

    H = H1[8]
    #print('\nH:\n',H)

    V12 = [H[0]*H[1], H[0]*H[4]+H[3]*H[1],
           H[3]*H[4], H[6]*H[1]+H[0]*H[7],
           H[6]*H[4]+H[3]*H[7], H[6]*H[7]]

    V11 = [H[0]*H[0], H[0]*H[3]+H[3]*H[0],
           H[3]*H[3], H[6]*H[0]+H[0]*H[6],
           H[6]*H[3]+H[3]*H[6], H[6]*H[6]]

    V22 = [H[1]*H[1], H[1]*H[4]+H[4]*H[1],
           H[4]*H[4], H[7]*H[1]+H[1]*H[7],
           H[7]*H[4]+H[4]*H[7], H[7]*H[7]]

    Vaux0=[V11[0]-V22[0], V11[1]-V22[1],V11[2]-V22[2],V11[3]-V22[3],V11[4]-V22[4],V11[5]-V22[5]]

    V121, V11, V22, H11= calculateV(points3D1, points2D1)
    Vaux1=[V11[0]-V22[0], V11[1]-V22[1],V11[2]-V22[2],V11[3]-V22[3],V11[4]-V22[4],V11[5]-V22[5]]
    
    V122, V11, V22, H22= calculateV(points3D2, points2D2)
    Vaux2=[V11[0]-V22[0], V11[1]-V22[1],V11[2]-V22[2],V11[3]-V22[3],V11[4]-V22[4],V11[5]-V22[5]]
    
    V=np.array([[V12],[Vaux0],[V121],[Vaux1],[V122],[Vaux2]])

    V= np.squeeze(V)
    #print('\nV:\n', V)

    U2, V2, S2 = np.linalg.svd(V)

    S=S2[5]
    #print('\nS:\n', S)

#--------------------Ixtrinsic_params---------------------------#
    c1=S[1]*S[3]-S[0]*S[4]
    c2=S[0]*S[2]-S[1]*S[1]
    v0=c1/c2
    #print('\nvO:',v0)
    landa=S[5]-(S[3]*S[3]+v0*c1)/S[0]

    alfau=np.sqrt(landa/S[0])

    alfav=np.sqrt(landa*S[0]/c2)
    s=-S[1]*alfau*alfau*alfav/landa
    u0= s*v0/alfau-S[3]*alfau*alfau/landa
    #print('\nuO:', u0)

    K = np.matrix([
        [alfau, s, u0],
        [0.0, alfav, v0],
        [0.0, 0.0, 1]])
    #print("\nIntrinsic_params:\n", K)

#--------------------Extrinsic_params Image0----------------------#
    h1 = np.array([H[0], H[3], H[6]])
    h1 = h1.reshape((3, 1))
    h2 = np.array([H[1], H[4], H[7]])
    h2 = h2.reshape((3, 1))
    h3 = np.array([H[2], H[5], H[8]])
    h3 = h3.reshape((3, 1))
    
    Kinv = np.linalg.inv(K)
    aux = Kinv*h1
    alfaAbs = (1/np.sqrt(aux[0]*aux[0]+aux[1]*aux[1]+aux[2]*aux[2]))[0,0]
    signAlfa=np.sign((Kinv*h3)[2])[0,0]
    alfa=alfaAbs*signAlfa
    r1=alfa*Kinv*h1
    r1=np.squeeze(r1)
    r2=alfa*Kinv*h2
    r2 = np.squeeze(r2)
    r3=np.cross(r1,r2)
    #print('\nr1 image0:', r1)
    #print('\nr2 image0:', r2)
    #print('\nr3 image0:', r3)
    T = alfa*Kinv*h3
    #print('\nT image0:\n', T)

    extrinsic_params = np.matrix([
        [r1[0, 0], r2[0, 0], r3[0, 0], T[0][0]],
        [r1[0, 1], r2[0, 1], r3[0, 1], T[1][0]],
        [r1[0, 2], r2[0, 2], r3[0, 2], T[2][0]]])

    #print("\nExtrinsic_params Image0:\n", extrinsic_params)
#--------------------Extrinsic_params Image1----------------------#
    h1 = np.array([H11[0], H11[3], H11[6]])
    h1 = h1.reshape((3, 1))
    h2 = np.array([H11[1], H11[4], H11[7]])
    h2 = h2.reshape((3, 1))
    h3 = np.array([H11[2], H11[5], H11[8]])
    h3 = h3.reshape((3, 1))

    Kinv = np.linalg.inv(K)
    aux = Kinv*h1
    alfaAbs = (1/np.sqrt(aux[0]*aux[0]+aux[1]*aux[1]+aux[2]*aux[2]))[0, 0]
    signAlfa = np.sign((Kinv*h3)[2])[0, 0]
    alfa = alfaAbs*signAlfa
    r1 = alfa*Kinv*h1
    r1 = np.squeeze(r1)
    r2 = alfa*Kinv*h2
    r2 = np.squeeze(r2)
    r3 = np.cross(r1, r2)
    #print('\nr1 image1:', r1)
    #print('\nr2 image1:', r2)
    #print('\nr3 image1:', r3)
    T = alfa*Kinv*h3
    #print('\nT image1:\n', T)

    extrinsic_params1 = np.matrix([
        [r1[0, 0], r2[0, 0], r3[0, 0], T[0][0]],
        [r1[0, 1], r2[0, 1], r3[0, 1], T[1][0]],
        [r1[0, 2], r2[0, 2], r3[0, 2], T[2][0]]])

    #print("\nExtrinsic_params Image1:\n", extrinsic_params1)
#--------------------Extrinsic_params Image2----------------------#
    h1 = np.array([H22[0], H22[3], H22[6]])
    h1 = h1.reshape((3, 1))
    h2 = np.array([H22[1], H22[4], H22[7]])
    h2 = h2.reshape((3, 1))
    h3 = np.array([H22[2], H22[5], H22[8]])
    h3 = h3.reshape((3, 1))

    Kinv = np.linalg.inv(K)
    aux = Kinv*h1
    alfaAbs = (1/np.sqrt(aux[0]*aux[0]+aux[1]*aux[1]+aux[2]*aux[2]))[0, 0]
    signAlfa = np.sign((Kinv*h3)[2])[0, 0]
    alfa = alfaAbs*signAlfa
    r1 = alfa*Kinv*h1
    r1 = np.squeeze(r1)
    r2 = alfa*Kinv*h2
    r2 = np.squeeze(r2)
    r3 = np.cross(r1, r2)
    #print('\nr1 image2:',r1)
    #print('\nr2 image2:', r2)
    #print('\nr3 image2:', r3)
    T = alfa*Kinv*h3
    #print('\nT image2:\n', T)

    extrinsic_params2 = np.matrix([
        [r1[0, 0], r2[0, 0], r3[0, 0], T[0][0]],
        [r1[0, 1], r2[0, 1], r3[0, 1], T[1][0]],
        [r1[0, 2], r2[0, 2], r3[0, 2], T[2][0]]])

    #print("\nExtrinsic_params Image2:\n", extrinsic_params2)

    return extrinsic_params, extrinsic_params1, extrinsic_params2, K


def calculaError(points3D0, points2D0, points3D1, points2D1, points3D2, points2D2, extrinsic, extrinsic1, extrinsic2, K):

    points3D0H = []
    points3D1H = []
    points3D2H = []
    for point in points3D0:
        points3D0H.append([point[0], point[1], point[2], 1])
    for point in points3D1:
        points3D1H.append([point[0], point[1], point[2], 1])
    for point in points3D2:
        points3D2H.append([point[0], point[1], point[2], 1])
    
    points3D0H = np.asarray(points3D0H)
    points3D1H = np.asarray(points3D1H)
    points3D2H = np.asarray(points3D2H)

    M0 = np.matmul(K,extrinsic)
    M1 = np.matmul(K,extrinsic1)
    M2 = np.matmul(K, extrinsic2)

    proj0 = np.matmul(M0, np.transpose(points3D0H))
    proj1 = np.matmul(M1, np.transpose(points3D1H))
    proj2 = np.matmul(M2, np.transpose(points3D2H))

    proj0 = np.transpose(proj0)
    proj1 = np.transpose(proj1)
    proj2 = np.transpose(proj2)

    error0=0
    i=0
    for point in proj0:

        aux0= point[0,0]/point[0,2]
        aux1 = point[0,1]/point[0,2]
        point_proje=np.array([aux0,aux1])
        point_known = np.array([points2D0[i][0], points2D0[i][1]])
        error_single = np.array([point_proje[0] - point_known[0],
                        point_proje[1] - point_known[1]])
        modulo = np.sqrt(
            error_single[0]*error_single[0]+error_single[1]*error_single[1])**2
        error0= error0+modulo
        i=i+1
    error0= error0/len(points2D0)

    error1 = 0
    i = 0
    for point in proj1:
        aux0 = point[0,0]/point[0,2]
        aux1 = point[0,1]/point[0,2]
        point_proje = np.array([aux0, aux1])
        point_known = np.array([points2D1[i][0], points2D1[i][1]])
        error_single = np.array([point_proje[0] - point_known[0],
                                 point_proje[1] - point_known[1]])
        modulo = np.sqrt(
            error_single[0]*error_single[0]+error_single[1]*error_single[1])**2
        error1 = error1+modulo
        i = i+1
    error1 = error1/len(points2D1)

    error2 = 0
    i = 0
    for point in proj2:
        aux0 = point[0,0]/point[0,2]
        aux1 = point[0,1]/point[0,2]
        point_proje = np.array([aux0, aux1])
        point_known = np.array([points2D2[i][0], points2D2[i][1]])
        error_single = np.array([point_proje[0] - point_known[0],
                                 point_proje[1] - point_known[1]])
        modulo = np.sqrt(
            error_single[0]*error_single[0]+error_single[1]*error_single[1])**2
        error2 = error2+modulo
        i = i+1
    error2 = error2/len(points2D2)

    return error0, error1, error2

#----------------------RANSAC----------------------#

def RANSAC(points3D0, points2D0, points3D1, points2D1, points3D2, points2D2, extrinsic, extrinsic1, extrinsic2, K, n, d, k):
    Ms0 = []
    Ms1 = []
    Ms2 = []

    M0_n = []
    M1_n = []
    M2_n = []


    MSE0 =[]
    MSE1 = []
    MSE2 = []

    MSEs0 = []
    MSEs1 = []
    MSEs2 = []

    inliers3D0 = []
    inliers2D0 = []
    inliers3D1 = []
    inliers2D1 = []
    inliers3D2 = []
    inliers2D2 = []

    for i in range (0, k):
        points3D0_random = []
        points2D0_random = []
        points3D1_random = []
        points2D1_random = []
        points3D2_random = []
        points2D2_random = []

        distances0 = []
        distances1 = []
        distances2 = []
       
        for i in range(0, n):
            random.seed()
            random_int0 = random.randint(0, len(points3D0)-1)
            random_int1 = random.randint(0, len(points3D1)-1)
            random_int2 = random.randint(0, len(points3D2)-1)

            points3D0_random.append(points3D0[random_int0])
            points2D0_random.append(points2D0[random_int0])

            points3D1_random.append(points3D1[random_int1])
            points2D1_random.append(points2D1[random_int1])

            points3D2_random.append(points3D2[random_int2])
            points2D2_random.append(points2D2[random_int2])

        extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n = calculaparameters(
            points3D0_random, points2D0_random, points3D1_random, points2D1_random, points3D2_random, points2D2_random)
        
#-------------Image0-------------#
        for i in range(0, len(points3D0_random)):
            distance = math.sqrt(calculaError(points3D0_random, points2D0_random, points3D1_random, points2D1_random,
                         points3D2_random, points2D2_random, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[0])
            distances0.append(distance)
        distances0.sort()

        if len(distances0)%2 == 0:
            median = (distances0[int(len(distances0)/2)]+distances0[(int(len(distances0)/2)-1)])/2
        else:
            median = distances0[int((len(distances0)/2)-0.5)]
        t = 1.5*median

        for i in range(0, len(points3D0)):
            distance = math.sqrt(calculaError(points3D0, points2D0, points3D1, points2D1,
                                              points3D2, points2D2, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[0])
            if distance < t:
                inliers3D0.append(points3D0[i])
                inliers2D0.append(points2D0[i])

        if len(inliers3D0) >= d:
            extrinsic0_n = calculaparameters(
                inliers3D0, inliers2D0, points3D1, points2D1, points3D2, points2D2)[0]
            MSE0 = math.sqrt(calculaError(inliers3D0, inliers2D0, points3D1, points2D1,
                               points3D2, points2D2, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[0])
            M0_n = np.matmul(K_n, extrinsic0_n)
            #print("\nM0_RANSAC\n", M0_n)
            #print("\n MSE Image 0 RANSAC:", MSE0)

        Ms0.append(M0_n)
        MSEs0.append(MSE0)
#-------------Image1-------------#
        for i in range(0, len(points3D1_random)):
            distance = math.sqrt(calculaError(points3D0_random, points2D0_random, points3D1_random, points2D1_random,
                                    points3D2_random, points2D2_random, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[1])
            distances1.append(distance)
        distances1.sort()

        if len(distances1) % 2 == 0:
            median = (distances1[int(len(distances1)/2)] +
                      distances1[(int(len(distances1)/2)-1)])/2
        else:
            median = distances1[int((len(distances1)/2)-0.5)]
        t = 1.5*median

        for i in range(0, len(points3D1)):
            distance = math.sqrt(calculaError(points3D0, points2D0, points3D1, points2D1,
                                              points3D2, points2D2, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[1])
            if distance < t:
                inliers3D1.append(points3D1[i])
                inliers2D1.append(points2D1[i])

        if len(inliers3D1) >= d:
            extrinsic1_n = calculaparameters(
                points3D0, points2D0, inliers3D1, inliers2D1, points3D2, points2D2)[1]
            MSE1 = math.sqrt(calculaError(points3D0, points2D0, inliers3D1, inliers2D1,
                                points3D2, points2D2, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[1])
            M1_n = np.matmul(K_n, extrinsic1_n)
            #print("\nM1_RANSAC\n", M1_n)
            #print("\n MSE Image 1 RANSAC:", MSE1)
        
        Ms1.append(M1_n)
        MSEs1.append(MSE1)
#---------------Image2------------#
        for i in range(0, len(points3D2_random)):
            distance = math.sqrt(calculaError(points3D0_random, points2D0_random, points3D1_random, points2D1_random,
                                    points3D2_random, points2D2_random, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[2])
            distances2.append(distance)
        distances2.sort()

        if len(distances2) % 2 == 0:
            median = (distances2[int(len(distances2)/2)] +
                      distances2[(int(len(distances2)/2)-1)])/2
        else:
            median = distances2[int((len(distances2)/2)-0.5)]
        t = 1.5*median

        for i in range(0, len(points3D2)):
            distance = math.sqrt(calculaError(points3D0, points2D0, points3D1, points2D1,
                                              points3D2, points2D2, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[2])
            if distance < t:
                inliers3D2.append(points3D2[i])
                inliers2D2.append(points2D2[i])

        if len(inliers3D2) >= d:
            extrinsic2_n = calculaparameters(
                points3D0, points2D0, points3D1, points2D1, inliers3D2, inliers2D2)[2]
            MSE2 = math.sqrt(calculaError(points3D0, points2D0, points3D1, points2D1,
                                inliers3D2, inliers2D2, extrinsic0_n, extrinsic1_n, extrinsic2_n, K_n)[2])
            M2_n = np.matmul(K_n, extrinsic2_n)
            #print("\nM2_RANSAC\n", M2_n)
            #print("\n MSE Image 2 RANSAC:", MSE2)

        Ms2.append(M2_n)
        MSEs2.append(MSE2)

    print("\nMs0", Ms0)
    print("\nMSEs0", MSEs0)

    print("\nMs1", Ms1)
    print("\nMSEs1", MSEs1)

    print("\nMs2", Ms2)
    print("\nMSEs2", MSEs2)
    
    MSEmin0 = MSEs0[0]
    Mdef0 = Ms0[0]
    for i in range(0, len(MSEs0)):
        if MSEs0[i] < MSEmin0:
            MSEmin0 = MSEs0[i]
            Mdef0 = Ms0[i]
    MSEmin1 = MSEs1[0]
    Mdef1 = Ms1[0]

    for i in range(0, len(MSEs1)):
        if MSEs1[i] < MSEmin1:
            MSEmin1 = MSEs1[i]
            Mdef1 = Ms1[i]

    MSEmin2 = MSEs2[0]

    Mdef2 = Ms2[0]
    for i in range(0, len(MSEs2)):
        if MSEs2[i] < MSEmin2:
            MSEmin2 = MSEs2[i]
            Mdef2 = Ms2[i]

    return Mdef0, MSEmin0, Mdef1, MSEmin1, Mdef2, MSEmin2
#------------------Calibration using Opencv funtion------------#
def calibrationOpenCV(f, g, img):
    
    points3D = []
    points3D.append(f)

    points2D = []
    g = np.expand_dims(g, axis=1)
    points2D.append(g)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    #calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        points3D, points2D, gray.shape[::-1], None, None)
    print('\nMatrix M with OpenCV of image 1:\n', mtx)
    #undistorsion
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('../data/calibresult.png', dst)

    mean_error = 0
    for i in range(len(points3D)):
        points2D2, _ = cv2.projectPoints(
            points3D[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(points2D[i], points2D2, cv2.NORM_L2)/len(points2D2)
        mean_error = mean_error + error

    print ("\nTotal error: ", mean_error/len(points3D))


def main():

    if len(sys.argv)!=5:
        print("Wrong number of arguments\n")
        print("It should be by the form:\n")
        print("python hw5_2.py file3d1.txt file3d2.txt file3d2.txt 3dpoints.txt\n")
    else:
        f = pd.read_csv("../data/points/3D/"+sys.argv[4],
                        delimiter=',', dtype=np.float32)
        f = np.asarray(f, dtype=np.float32)
        g = pd.read_csv("../data/points/2D/"+sys.argv[1], delimiter=',', dtype=np.float32)
        g = np.asarray(g, dtype=np.float32)
        
        f1 = pd.read_csv("../data/points/3D/"+sys.argv[4],
                         delimiter=',', dtype=np.float32)
        f1 = np.asarray(f1, dtype=np.float32)
        g1 = pd.read_csv("../data/points/2D/"+sys.argv[2], delimiter=',', dtype=np.float32)
        g1 = np.asarray(g1, dtype=np.float32)
        
        f2 = pd.read_csv("../data/points/3D/"+sys.argv[4],
                         delimiter=',', dtype=np.float32)
        f2 = np.asarray(f2, dtype=np.float32)
        g2 = pd.read_csv("../data/points/2D/"+sys.argv[3], delimiter=',', dtype=np.float32)
        g2 = np.asarray(g2, dtype=np.float32)
        
        extrinsic, extrinsic1, extrinsic2, K= calculaparameters(f,g,f1,g1,f2,g2)
        print("\nINTRINSIC_params:\n", K)
        print("\nExtrinsic_params Image0:\n", extrinsic)
        print("\nExtrinsic_params Image1:\n", extrinsic1)
        print("\nExtrinsic_params Image2:\n", extrinsic2)

        error1 ,error2 ,error3= calculaError(f, g, f1, g1, f2, g2, extrinsic, extrinsic1, extrinsic2, K)
        
        print('\nError Image0:', error1)
        print('\nError Image1:', error2)
        print('\nError Image2:', error3)
        
        ransac_params = pd.read_csv("../data/RANSAC.config", delimiter=',')

        ransac_params = np.asarray(ransac_params)
        ransac_params = np.squeeze(ransac_params)

        Mdef0, MSEmin0, Mdef1, MSEmin1, Mdef2, MSEmin2 = RANSAC(
            f, g, f1, g1, f2, g2, extrinsic, extrinsic1, extrinsic2, K, ransac_params[0], ransac_params[1], ransac_params[2])
        
        print("\nM0 from Ransac\n", Mdef0)
        print("\nMSE0 from RANSAC", MSEmin0)
        print("\nM1 from Ransac\n", Mdef1)
        print("\nMSE1 from RANSAC\n", MSEmin1)
        print("\nM2 from Ransac\n", Mdef2)
        print("\nMSE2 from RANSAC", MSEmin2)
        
        img = cv2.imread('../data/chess1.jpg')
        calibrationOpenCV(f,g,img)
    

if __name__ == '__main__':
    main()
