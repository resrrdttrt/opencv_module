from tkinter import TRUE
import cv2
import numpy as np
import Module.StackImage as SI

def nothing(x):
    pass


def main():
    img =cv2.imread('Image/hinh_hoc.png')
    # img =cv2.imread('Image/hinh_vuong.jpg')
    # img=cv2.imread('Image/hinh.jpg')
    cv2.namedWindow('Threshold')
    cv2.resizeWindow('Threshold',500,300)
    cv2.createTrackbar('Blur','Threshold',0,10,nothing)
    cv2.createTrackbar('Threshold min','Threshold',0,255,nothing)
    cv2.createTrackbar('Threshold max','Threshold',75,255,nothing)
    cv2.createTrackbar('Epsilon','Threshold',10,100,nothing)
    cv2.createTrackbar('Area','Threshold',500,5000,nothing)
    cv2.createTrackbar('Scale','Threshold',50,100,nothing)


    while True:
        imgapprox=img.copy()
        imgCnt = img.copy()
        # blank=np.zeros(img.shape,dtype=np.uint8)
        # blank[:]=(255,255,255)
        threshold_min=cv2.getTrackbarPos('Threshold min','Threshold')
        # threshold_max=cv2.getTrackbarPos('Threshold max','Threshold')
        b=cv2.getTrackbarPos('Blur','Threshold')*2+1
        blur=cv2.GaussianBlur(img,(b,b),cv2.BORDER_DEFAULT)
        gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        # threshold=cv2.adaptiveThreshold(gray,threshold_min,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,2) #chon threshold tu dong dua theo lan can gauss_c hoac mean_c

        ret,threshold=cv2.threshold(gray,threshold_min,255,cv2.THRESH_BINARY) #x>125 thi x=175;x<125 thi x=0
        contours,hierarchie=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


        # canny=cv2.Canny(gray,threshold_min,threshold_max)
        # dilated=cv2.dilate(canny,(5,5),iterations=2) #=1 neu it nhat 1 phan tu trong Kernel=1
        # contours,hierarchie=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area>cv2.getTrackbarPos('Area','Threshold'):
                cv2.drawContours(imgCnt,cnt,-1,(0,0,0),6)
                epsilon=cv2.getTrackbarPos('Epsilon','Threshold')/1000*cv2.arcLength(cnt,True)
                approx=cv2.approxPolyDP(cnt,epsilon,True)
                x,y,w,h=cv2.boundingRect(cnt)
                cv2.rectangle(imgapprox,(x,y),((x+w),(y+h)),(0,0,0),6)
                cv2.putText(imgapprox,f'{len(approx)} points',(x+5,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                cv2.putText(imgapprox,f'{area} points',(x+5,y+h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        scale=cv2.getTrackbarPos('Scale','Threshold')/100
        stack=SI.stackImage(scale,[[img,blur,gray],[threshold,imgCnt,imgapprox]])

        # stack=SI.stackImage(scale,[[img,blur,canny],[dilated,imgCnt,imgapprox]])
        cv2.imshow('Stack',stack)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


    cv2.destroyAllWindows()




def shape_detect_canny(img,bkernel,threshold_min,threshold_max,exactness_min,area_min): #better
    imgapprox=img.copy()
    blur=cv2.GaussianBlur(img,(bkernel,bkernel),cv2.BORDER_DEFAULT)
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

    canny=cv2.Canny(gray,threshold_min,threshold_max)
    dilated=cv2.dilate(canny,(5,5),iterations=2)
    contours,hierarchie=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>area_min:
            epsilon=exactness_min*cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,epsilon,True)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(imgapprox,(x,y),((x+w),(y+h)),(0,0,0),6)
            if len(approx)==3:
                shape='Triangel'
            elif len(approx)==4:
                shape='Rectangle'
            elif len(approx)==5:
                shape='Paragon'
            elif len(approx)>5:
                shape='Round'
            else:
                shape='None'
            cv2.putText(imgapprox,f'{shape} {len(approx)}',(x+5,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
           

    return imgapprox

def shape_detect_threshold(img,bkernel,threshold_min,exactness_min,area_min):
    imgapprox=img.copy()
    blur=cv2.GaussianBlur(img,(bkernel,bkernel),cv2.BORDER_DEFAULT)
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

    ret,threshold=cv2.threshold(gray,threshold_min,255,cv2.THRESH_BINARY) #x>125 thi x=175;x<125 thi x=0
    contours,hierarchie=cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>area_min:
            epsilon=exactness_min*cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,epsilon,True)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(imgapprox,(x,y),((x+w),(y+h)),(0,0,0),6)
            if len(approx)==3:
                shape='Triangel'
            elif len(approx)==4:
                shape='Rectangle'
            elif len(approx)==5:
                shape='Paragon'
            else:
                shape='Round'
            cv2.putText(imgapprox,f'{shape} {len(approx)}',(x+5,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
           


if __name__ == '__main__':
    main()