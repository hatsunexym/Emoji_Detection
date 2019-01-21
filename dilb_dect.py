import sys  
import os  
import dlib  
import glob
import cv2
from skimage import io  
cap = cv2.VideoCapture(0)
#'rtsp://admin:Chen1qaz@192.168.1.108/cam/realmonitor?channel=1&subtype=0'
n=-1
detector = dlib.get_frontal_face_detector()  
while (1) : 
    #cv2.imshow("Oto Video", frame)
    #videoWriter.write(frame)
    n=n+1
    if n%15!=0:
        continue
    ret, im = cap.read()
    cv2.imwrite("temp_dlib.jpg", im)
        
    #predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  
    #win = dlib.image_window()  

    img = io.imread("temp_dlib.jpg")  
    #win.clear_overlay()  
    #win.set_image(img)  
        
    dets = detector(img, 1)  
    print("Number of faces detected: {}".format(len(dets)))  
    for k, d in enumerate(dets):  
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(  
        #    k, d.left(), d.top(), d.right(), d.bottom())) 
        cv2.rectangle(im, (int(d.left()),int(d.bottom())), (int(d.right()),int(d.top())), (0,255,0))
    
        
        
    cv2.imshow("capture", im)
    cv2.waitKey(50) 
 
    if n==150000:
        n=0
  
