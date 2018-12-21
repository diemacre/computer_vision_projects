import cv2
import numpy as np
import sys




def main():
    points3D = []
    points2D = []
    
    if len(sys.argv)!=2:
        print("Wrong arguments, write as argument the name of the image located in ../data/ folder")
    else:
        print('../data/'+sys.argv[1])
        image = cv2.imread('../data/'+sys.argv[1])
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image_gray,(7,6),None)
    
        if ret == True:
            criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            coords = np.zeros((6*7, 3), np.float32)
            coords[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
            points3D = np.squeeze(coords)
    
            corners_new = cv2.cornerSubPix(image_gray,corners,(11,11),(-1,-1),criteria)
            points2D = np.squeeze(corners_new)
            cv2.drawChessboardCorners(image, (7,6), corners_new,ret)
            cv2.imshow('image',image)
        
            argv1 = sys.argv[1].replace('.jpg', '')

            f = open("../data/points/2D/"+argv1+".txt","w")
            for i in range(len(points2D)):
                f.write(str(points2D[i][0])+','+str(points2D[i][1])+'\n')
            print('2D points saved in 2D.txt file.\n')
            f.close()

            g = open("../data/points/3D/world.txt","w")
            for i in range(len(points3D)):
                g.write(str(points3D[i][0])+',' +
                        str(points3D[i][1])+','+str(points3D[i][2])+'\n')
            print('3D points saved in 3D.txt file.\n')
            g.close()

            key = cv2.waitKey()

            if key == 27:
                cv2.destroyAllWindows()
                print('Program closed, pressed "Esc" key')
        else:
            print("Points of the image have not been found. or something went wrong.")

if __name__ == '__main__':
    main()
