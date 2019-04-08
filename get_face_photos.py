#-*- coding:utf-8 -*-


import os
import random
import numpy as np
import cv2
import PIL

IMG_SIZE = 64


def create_dir(*args):
    #create dir
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

def get_padding_size(shape):
    #make square rect pic
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int)) - np.array([h,h,w,w], int) //2
    return list(result)


def deal_with_iamge(img, h=64, w=64):
    """image width to 64*64"""
    top, bottom, left, right = get_padding_size(img.shape[0:2])
    img = cv2.copyMakeBorder()
    img = cv2.resize(img, (h, w))
    return img

#明暗处理
def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(np.float32)
    imgsrc = imgsrc * alpha + bias

    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def get_face_from_camera(outdir):
    create_dir(outdir)
    camera = cv2.VideoCapture(0)
    #使用人脸识别分类器
    haar = cv2.CascadeClassifier('/home/alex/opencv-4.0.1/data/haarcascades/haarcascade_frontalface_default.xml')
    n = 1
    while True:
        if (n <= 200):
            print('It`s processing %s image.' % n)
            # 读帧
            success, img = camera.read()
            # to grayscale pic
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #识别人脸部分，缩放比和有效点数
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                #could deal with face to train
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                cv2.putText(img, 'wujinlong', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # add name
                #用框包起来人脸
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n+=1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    name = input('please input yourename: ')
    get_face_from_camera(os.path.join('./image/trainfaces/', name))
    print(os.path.join('./image/trainfaces/', name))