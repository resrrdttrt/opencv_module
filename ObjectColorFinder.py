import cv2
import numpy as np

def nothing(x):
    pass

def main():
    pass

def find_color(img):
    img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.namedWindow('Threshold')
    cv2.resizeWindow('Threshold',600,350)

    cv2.createTrackbar('Hmax','Threshold',180,180,nothing)
    cv2.createTrackbar('Hmin','Threshold',0,180,nothing)
    cv2.createTrackbar('Smax','Threshold',255,255,nothing)
    cv2.createTrackbar('Smin','Threshold',0,255,nothing)
    cv2.createTrackbar('Vmax','Threshold',255,255,nothing)
    cv2.createTrackbar('Vmin','Threshold',0,255,nothing)
    while True:
        hmax = cv2.getTrackbarPos('Hmax','Threshold')
        hmin = cv2.getTrackbarPos('Hmin','Threshold')
        smax = cv2.getTrackbarPos('Smax','Threshold')
        smin = cv2.getTrackbarPos('Smin','Threshold')
        vmax = cv2.getTrackbarPos('Vmax','Threshold')
        vmin = cv2.getTrackbarPos('Vmin','Threshold')
        upper=np.array([hmax,smax,vmax])
        lower=np.array([hmin,smin,vmin])
        mask=cv2.inRange(img_HSV,lower,upper)
        result=cv2.bitwise_and(img,img,mask=mask)
        vstack=np.vstack([result,img])
        cv2.imshow('vstack',vstack)

        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()

