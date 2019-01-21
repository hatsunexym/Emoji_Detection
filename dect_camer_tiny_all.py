# -*- coding: utf-8 -*-  
from ctypes import *
import math
import random
import cv2
import argparse
import MySQLdb
import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import Image
import sys
import dlib
import glob
from skimage import io

t = time.time()

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/chen/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]




def faceRecognition(imagepath):


	#图片预处理设置

	im=caffe.io.load_image(img,False)#加载图片
	
	return face #face是最终识别的表情

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh, hier_thresh=.5, nms=.45):
    print ("22222222222222")
    thresh = float(thresh)
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='dection demo')
    #parser.add_argument('--rtsp', dest='rtsp')
    parser.add_argument('--sheld', dest='sheld')
    args = parser.parse_args()

    return args

    
if __name__ == "__main__":
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    #img = cv2.imread('1.jpg')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    #net = load_net("cfg/yolo.cfg","/home/wang/darknet/yolo.weights",int(0))
    db = MySQLdb.connect("192.168.1.102","root","123456","dect",charset = 'utf8')
    cursor = db.cursor()

    args = parse_args()

    #rtsp = args.rtsp
    sheld = args.sheld

    a=  c_char_p(b'cfg/tiny-yolo.cfg')
    b = c_char_p(b'/home/chen/darknet/yolo-tiny.weights')
    net = load_net(a,b, 0)
    c = c_char_p(b'cfg/coco.data')
    meta = load_meta(c)
    """
    videoCapture = cv2.VideoCapture(avi)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('q.mp4', cv2.VideoWriter_fourcc(*'X264'), 24, size) 
    success, im = videoCapture.read()
    n =0
    """
    id ='2'
    #videoWriter = cv2.VideoWriter('de.mp4', cv2.VideoWriter_fourcc(*'X264'), 1, (500,700))
    cap = cv2.VideoCapture('rtsp://admin:Chen1qaz@192.168.1.108/cam/realmonitor?channel=1&subtype=0')
    n=-1
    b = 0

    """
    age____init--------
    """



    caffe_root = '/hoem/chen/caffe/' 
    import sys
    sys.path.insert(0, caffe_root + 'python')
    import caffe


    mean_filename='age_model/mean.binaryproto'
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean  = caffe.io.blobproto_to_array(a)[0]

    age_net_pretrained='age_model/age_net.caffemodel'
    age_net_model_file='age_model/deploy_age.prototxt'
    age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                           mean=mean,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))


    gender_net_pretrained='age_model/gender_net.caffemodel'
    gender_net_model_file='age_model/deploy_gender.prototxt'
    gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                           mean=mean,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))


    age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    gender_list=['Male','Female']







    """
    biaoqing_init
    """

    root='/home/chen/caffe/'   #根目
    deploy=root + 'face_net/deploy.prototxt'    #deploy文件
    caffe_model=root + 'result/facenet/facialnet_solver_iter_200000.caffemodel'   #训练好的 caffemodel
   
    labels_filename = root + 'face_net/labels.txt'  #类别名称文件，将数字标签转换回类别名称
    net_b = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network
    transformer = caffe.io.Transformer({'data': net_b.blobs['data'].data.shape})  #设定图片的shape格式(1,1,42,42)
    transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(42,42,1)变为(1,42,42)
    #transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
    net_b.blobs['data'].reshape(1,1,42,42)

    detector = dlib.get_frontal_face_detector()
    while (1) : 
        #cv2.imshow("Oto Video", frame)
        #videoWriter.write(frame)
        ret, im = cap.read()
        n=n+1
        if n%60!=0:
            continue
        cv2.imwrite("temp.jpg", im)
        d = c_char_p(b'temp.jpg')
        #r = detect(net, meta, d)
        #videoWriter.write(frame)
        #success, frame = videoCapture.read()
        #n = n+1
        #d = c_char_p(b'person.jpg')
        #im = cv2.imread('person.jpg')
        r = detect(net, meta, d,sheld)
        print (r)
        exis_person ='0'
        num = 0
        b = b%6

        face_num=0
        #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        age_gender=[]
        img = io.imread("temp.jpg")
        dets = detector(img, 1)
        print ('wwwwwwwwwwwwwwwwwwwwwww')
        #for (x,y,w,h) in faces:
        for k, d in enumerate(dets):
            print ("-----is x---")
            print (dets)
            print (d.bottom(),int(d.top()),int(d.left()),int(d.right()))
            face_cut = im[int(d.top()):int(d.bottom()),int(d.left()):int(d.right())]
            #res2 = cv2.resize(face_cut,(42,42),interpolation=cv2.INTER_CUBIC)
            face_name = 'face_cut/face'+str(face_num)+'.jpg'
            cv2.imwrite(face_name, face_cut)
            input_image = caffe.io.load_image(face_name)
            prediction = age_net.predict([input_image]) 
            age = age_list[prediction[0].argmax()]
            print (age) 
            prediction = gender_net.predict([input_image]) 
            gender = gender_list[prediction[0].argmax()]
            print (gender)
            #face=faceRecognition(face_name)
            


            fa=caffe.io.load_image(face_name,False)
            net_b.blobs['data'].data[...] = transformer.preprocess('data',fa)#执行上面设置的图片预处理操作，并将图片载入到blob中

            #执行测试
            out = net_b.forward()
            labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
            prob= net_b.blobs['prob'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印
            order=prob.argsort()[-1]  #将概率值排序，取出最大值所在的序号 
            face=labels[order]
            sql4 ="delete from  person_attribute where person_id ="+str(face_num)
            cursor.execute(sql4)
            db.commit()
            #cv2.putText(im, age, (int(x+w), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, False)
            sql3 ="insert into  person_attribute(id,age,person_id,gender, Expression)value("+id+","+"'"+str(age)+"'"+","+str(face_num)+","+"'"+gender+"'"+","+"'"+face+"'"+")"
            print (sql3)
            cursor.execute(sql3)
            db.commit()
            face_num  = face_num+1
            
            age_gender.append([age,gender,face])

        print (age_gender)
        
        for i in range(len(r)):
            m = str(r[i]).split(',')
            lable = m[0][2:-1]
            acc = m[1]
            m[2]= m[2][1:]
            x1=m[2].strip('(')
            x1 = float(x1)
            y1 = float(m[3])
            x2 = float(m[4])
            m[5]=m[5].strip(')')
            print (m[5])
            y2 = m[5]
            y2 = float(y2)
            
            if lable == "person":
                cv2.rectangle(im, (int(x1-x2/2),int(y1-y2/2)), (int(x2/2+x1),int(y2/2+y1)), (0,255,0))
                cv2.putText(im, lable, (int(x1-x2/2), int(y1-y2/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, False)
                exis_person ='1'
                num=num+1
        cv2.putText(im, 'On-line number: '+str(num), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, False)
        person_id=0
        #for (x,y,w,h) in faces:
        for k, d in enumerate(dets):
            #cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.rectangle(im, (int(d.left()),int(d.bottom())), (int(d.right()),int(d.top())), (0,255,0))
            print (age_gender)
            print ("1111111111111111111111111")
            age = str(age_gender[person_id])
            cv2.putText(im, age, (int(d.right()), int(d.top())), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, False)
            person_id = person_id+1
            
            
        sql ="delete from  person_dect where id ="+id
        cursor.execute(sql)
        db.commit()
        sql2 ="insert into  person_dect(id,exis_person,num,videoname,videourl)value("+id+","+exis_person+","+str(num)+","+"'"+"video1"+"'"+","+"'"+"rtsp://admin:Chen1qaz@192.168.1.109/cam/realmonitor?channel=1&subtype=0"+"'"+")"
        print (sql2)
        cursor.execute(sql2)
        db.commit()
        res=cv2.resize(im,(1800,1000),interpolation=cv2.INTER_CUBIC)
        #cv2.imwrite('a.jpg',im)
        #videoWriter.write(im)
        #success, im = videoCapture.read()
        ti = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
        print (ti[-2:])
        name = '/home/chen/apache-tomcat-7.0.81/webapps/printscreen/1/'+ti[-2:]+'.jpg'
        b= b+1
        cv2.imwrite(name,im)
        cv2.imshow("capture", res)
        cv2.waitKey(67)
        #res=cv2.resize(im,(4096,2304),interpolation=cv2.INTER_CUBIC)
        #videoWriter.write(im)
        #name = '/home/chen/apache-tomcat-7.0.81/webapps/printscreen/'+id+'.jpg'
        #cv2.imwrite(name,res)
        #if n%1800==0:
        #    name = 'results/'+id+'.jpg'
        #    cv2.imwrite(name,im)
        if n==18000000:
            n=0
#f = open ("success.txt",'w')
#print >> f, 'success' 
#f.close()
    #cv2.imshow("Oto Video", im)
    #cv2.waitKey(1000)
"""
    videoCapture = cv2.VideoCapture('1.avi')

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('1_s.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size) 
    success, frame = videoCapture.read()
    while success : 
        #cv2.imshow("Oto Video", frame)
        #videoWriter.write(frame)
        cv2.imwrite("temp.jpg", frame)
        d = c_char_p(b'temp.jpg')
        r = detect(net, meta, d)
        videoWriter.write(frame)
        success, frame = videoCapture.read()
      
        print (r)

        print ('1111111111111111111111111111')
        print (r[1])
    

    success, frame = videoCapture.read()
    cv2.imshow("Oto Video", frame)
    while success:
        cv2.imshow('MyWindow', frame)
        cv2.waitKey(1000)
        success, frame = videoCapture.read()
"""
