import cv2
import numpy as np

def nothing(x): #callback function
    pass



def main():
    img = np.zeros((400,400,3), np.uint8)
    hsv_frame=np.zeros((400,400,3), np.uint8)

    cv2.namedWindow('Window')
    cv2.resizeWindow('Window',400,400)


    state=0

    cv2.createTrackbar('B','Window',0,255,nothing)
    cv2.createTrackbar('G','Window',0,255,nothing)
    cv2.createTrackbar('R','Window',0,255,nothing)
    cv2.createTrackbar('H','Window',0,180,nothing)
    cv2.createTrackbar('S','Window',0,255,nothing)
    cv2.createTrackbar('V','Window',0,255,nothing)
    cv2.createTrackbar('State','Window',0,1,nothing)

    while(1):
        if state==0:
            r = cv2.getTrackbarPos('R','Window')
            g = cv2.getTrackbarPos('G','Window')
            b = cv2.getTrackbarPos('B','Window')
            img[:] = [b,g,r]
            hsv_frame= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            cv2.setTrackbarPos('H','Window',hsv_frame[0][0][0])
            cv2.setTrackbarPos('S','Window',hsv_frame[0][0][1])
            cv2.setTrackbarPos('V','Window',hsv_frame[0][0][2])
        elif state==1:
            h = cv2.getTrackbarPos('H','Window')
            s = cv2.getTrackbarPos('S','Window')
            v = cv2.getTrackbarPos('V','Window')
            hsv_frame[:]=(h,s,v)
            img=cv2.cvtColor(hsv_frame,cv2.COLOR_HSV2BGR)
            cv2.setTrackbarPos('B','Window',img[0][0][0])
            cv2.setTrackbarPos('G','Window',img[0][0][1])
            cv2.setTrackbarPos('R','Window',img[0][0][2])        


        state=cv2.getTrackbarPos('State','Window')
        cv2.imshow('Window',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


    cv2.destroyAllWindows()

def BGR2HSV(b,g,r):
    img_BGR=np.zeros((1,1,3),dtype=np.uint8)
    img_BGR[:]=(b,g,r)
    img_HSV=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
    return img_HSV

def HSV2BGR(h,s,v):
    img_HSV=np.zeros((1,1,3),dtype=np.uint8)
    img_HSV[:]=(h,s,v)
    img_BGR=cv2.cvtColor(img_HSV,cv2.COLOR_BGR2HSV)
    return img_BGR

if __name__ == '__main__':
    main()