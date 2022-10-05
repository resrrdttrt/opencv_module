import cv2 as cv
import mediapipe as mp
import numpy as np
import time

class meshDetector():
    def __init__(self,mode=False,maxFaces=1,refine_landmarks=False,minDet=0.5,minTrack=0.5):
        self.static_image_mode=mode
        self.max_num_faces=maxFaces
        self.refine_landmarks=refine_landmarks
        self.min_detection_confidence=minDet
        self.min_tracking_confidence=minTrack

        self.mpMeshs=mp.solutions.face_mesh
        self.mpDraws=mp.solutions.drawing_utils
        self.meshs=self.mpMeshs.FaceMesh(self.static_image_mode,self.max_num_faces,self.refine_landmarks,self.min_detection_confidence,self.min_tracking_confidence)

    def find_landmark(self,img,draw=True):
        draw_spec=self.mpDraws.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=1)
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.result=self.meshs.process(imgRGB)
        if draw:
            if self.result.multi_face_landmarks: #list contain 468 points of mesh
                for value in self.result.multi_face_landmarks:
                    self.mpDraws.draw_landmarks(img,value,connections=self.mpMeshs.FACEMESH_CONTOURS,landmark_drawing_spec=draw_spec) #xem define ham draw_landmarks de cai dat lai
        return img


def main():
    vid=cv.VideoCapture(0)
    detector=meshDetector()
    while True:
        isTrue,unflip_frame=vid.read()
        frame=cv.flip(unflip_frame,1)
        frame=detector.find_landmark(frame,draw=True)
        cv.imshow('Camera',frame)
        if cv.waitKey(1)==27:
            break

    vid.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    main()