import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import math
import trainlist
import tensorflow as tf

image=tf.keras.preprocessing.image

# Load the model
model = load_model('./Model/sign_1.h5')

# CAMERA can be 0 or 1 based on default camera of your computer.
cap = cv.VideoCapture(0)

# Grab the labels from the labels.txt file. This will be used later.
labels = open('./Model/labels.txt', 'r').readlines()
detector = HandDetector(maxHands=1)
offset=20
img_size=300
counter=0

while True:
    ret,img=cap.read()
    img_out=img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        #Image empty
        img_bg=np.ones((img_size,img_size,3), np.uint8)*255
        croped_img=img[y-offset:y+ h+offset,x-offset:x+ w+offset]
          
        aspect_ratio=h/w
       
        if aspect_ratio>1:
            k=img_size/h
            wCal= math.ceil(k*w)
            img_resize=cv.resize(croped_img,(wCal,img_size))
            wGap =math.ceil((img_size-wCal)/2)
            img_bg[:,wGap:wCal+wGap] = img_resize
            mg=image.load_img(img_bg,target_size=(224,224))
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            pred=np.argmax(model.predict(x))
            op=trainlist.dataset
            ans=op[pred]
            print("\n\n"+ans)
            
        else:
            k=img_size/w
            hCal= math.ceil(k*h)
            img_resize=cv.resize(croped_img,(img_size,hCal))
            hGap =math.ceil((img_size-hCal)/2)
            img_bg[hGap:hCal+hGap,:] = img_resize
            mg=image.load_img(img_bg,target_size=(224,224))
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            pred=np.argmax(model.predict(x))
            op=trainlist.dataset
            ans=op[pred]
            print("\n\n"+ans)
            
        cv.putText(img_out,labels[pred],(x,y-20),cv.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
            
        cv.imshow("Image_croped",croped_img)
        img_bw=cv.cvtColor(img_bg,cv.COLOR_BGR2GRAY)
        cv.imshow("Image_bg",img_bw)
            
        gesture=labels[pred]
        count+=1
        if count==30:
            if gesture!=list[-1]:
                list.append(gesture)
            count=count-30
        print(list)
    cv.imshow("Image",img_out)    
    # Listen to the keyboard for presses.
    keyboard_input = cv.waitKey(1)
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == ord('q'):
        break

cap.release()
cv.destroyAllWindows()