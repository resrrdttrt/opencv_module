import cv2 as cv
import numpy as np
import os



def object_detect(img1,img2,threshold):
    orb=cv.ORB_create(nfeatures=500)
    kp1,des1=orb.detectAndCompute(img1,None)
    kp2,des2=orb.detectAndCompute(img2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=False)
    try:
        match=bf.knnMatch(des1,des2,k=2) #trả về k list tốt nhât để kiểm tra ratio
        good = []
        for m,n in match:
            if m.distance < 0.75*n.distance:
                good.append(m)
    except:
        pass
    return(len(good))





