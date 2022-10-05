import cv2 as cv
import mediapipe as mp
import time

class faceDetector():
    def __init__(self,minDet=0.5,model=0):
        self.min_detection_confidence=minDet
        self.model_selection=model

        self.mpFaces=mp.solutions.face_detection
        self.mpDraws=mp.solutions.drawing_utils
        self.faces=self.mpFaces.FaceDetection(self.min_detection_confidence,self.model_selection)

    def find_landmark(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.result=self.faces.process(imgRGB)
        if draw:
            if self.result.detections: # tra ve 1 list chua toa do cua bounding box va toa do cua 6 diem tren mat 
                for value in self.result.detections:
                    self.mpDraws.draw_detection(img,value)
        return img

    def find_position(self,img,face_number):
        lmlisk=[]
        if self.result.detections:
            myface=self.result.detections[face_number].location_data
            for id,detection in enumerate(myface.relative_keypoints):
                h,w,c=img.shape
                cx,cy=int(detection.xmin*w),int(detection.ymin*h)
                lmlisk.append([id,cx,cy])
        return lmlisk

    def find_all_bbox(self,img):
        self.boxList=[]
        if self.result.detections:
            for id,detection in enumerate(self.result.detections):
                bboxC=detection.location_data.relative_bounding_box
                h,w,c=img.shape
                bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
                cv.rectangle(img,bbox,(255,0,0),1)
                #fancy draw
                cv.line(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2]//4,bbox[1]),(255,0,0),3)
                cv.line(img,(bbox[0],bbox[1]),(bbox[0],bbox[1]+bbox[2]//4),(255,0,0),3)

                cv.line(img,(bbox[0],bbox[1]+bbox[2]),(bbox[0]+bbox[2]//4,bbox[1]+bbox[2]),(255,0,0),3)
                cv.line(img,(bbox[0],bbox[1]+bbox[2]),(bbox[0],bbox[1]+bbox[2]-bbox[2]//4),(255,0,0),3)

                cv.line(img,(bbox[0]+bbox[2],bbox[1]),(bbox[0]+bbox[2]-bbox[2]//4,bbox[1]),(255,0,0),3)
                cv.line(img,(bbox[0]+bbox[2],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[2]//4),(255,0,0),3)

                cv.line(img,(bbox[0]+bbox[2],bbox[1]+bbox[2]),(bbox[0]+bbox[2],bbox[1]+bbox[2]-bbox[2]//4),(255,0,0),3)
                cv.line(img,(bbox[0]+bbox[2],bbox[1]+bbox[2]),(bbox[0]+bbox[2]-bbox[2]//4,bbox[1]+bbox[2]),(255,0,0),3)

                cv.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-10),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),2)
                self.boxList.append(bbox)
        return img



def main():
    vid=cv.VideoCapture(0)
    detector=faceDetector()
    preTime=0
    while True:
        isTrue,unflip_frame=vid.read()
        frame=cv.flip(unflip_frame,1)
        curTime = time.time()
        fps = int(1 / (curTime - preTime))
        preTime = curTime
        cv.putText(frame, f'FPS:{fps}', (20, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        frame=detector.find_landmark(frame,draw=False)
        frame=detector.find_all_bbox(frame)
        cv.imshow('Camera',frame)
        # for id,detection in enumerate(detector.result.detections):
        #     print(id,detection)
        # print(detector.result.detections[0].location_data.relative_keypoints[0])
        if cv.waitKey(1)==27:
            break

    vid.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    main()