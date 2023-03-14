#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install imutils


# In[43]:


from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial import distance as dist
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np


# In[46]:


train = pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
image_name_arr = train["image_name"].values


# In[55]:


fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(15,15), gridspec_kw={'wspace':0.1, 'hspace':0})
j=0
ran = 4352
for i in range(ran+6):
  if i>ran-1:
    im = cv2.imread("../input/jpeg-melanoma-256x256/train/"+image_name_arr[i]+".jpg")
    im = HAIR_SORRY_REMOVE(im)
    im = detecting_nevus(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax[j].imshow(im)
    ax[j].axis("off")
    j+=1


# In[48]:


def HAIR_SORRY_REMOVE(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    return (final_image)


# In[49]:


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def detecting_nevus(img_start):
  # load img and Blur
  img_start2 = cv2.GaussianBlur(img_start, ( 17, 17 ), 0)
  # img_start2 = cv2.blur(img_start,(10,10))
  Z = img_start2.reshape((-1,3))
  Z = np.float32(Z)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 2
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  thresh = center[label.flatten()]
  thresh = thresh.reshape((img_start.shape))
  # plt.imshow(img)

  thresh = cv2.cvtColor( thresh, cv2.COLOR_BGR2GRAY)

  thresh = cv2.Canny( thresh, 50, 60)
  kernel = np.ones((3,3),np.uint8)
  thresh = cv2.dilate( thresh, kernel, iterations=1)


  contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  im2 = cv2.drawContours(img_start.copy(), contours,-1, (0, 255, 0), 3)
  orig = img_start.copy()
  S = list()
  for c in contours:
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box)  if imutils.is_cv2() else cv2.boxPoints( box)
    box = np.array(box, dtype="int")
    box = perspective.order_points( box)
    center = 256/2
    delta = 70
    ( tl, tr, br, bl) = box
    (centerX, centerY) = midpoint( tl, br)
    if (centerX>center-delta and centerX<center+delta) and (centerY>center-delta and centerY<center+delta):
          if cv2.contourArea(c) >  70 and cv2.contourArea(c) <  20000:
            S.append(cv2.contourArea(c))
  if len(S)==0:
    for c in contours:
      box = cv2.minAreaRect(c)
      box = cv2.cv.BoxPoints(box)  if imutils.is_cv2() else cv2.boxPoints( box)
      box = np.array(box, dtype="int")
      box = perspective.order_points( box)
      center = 256/2
      delta = 60
      ( tl, tr, br, bl) = box
      (centerX, centerY) = midpoint( tl, br)
      if (centerX>center-delta and centerX<center+delta) and (centerY>center-delta and centerY<center+delta):
        if cv2.contourArea(c) >  40 and cv2.contourArea(c) <  20000:
          S.append(cv2.contourArea(c))
  if len(S) == 0:
     return img_start
    
  if len(S)>0:
    for c in contours:

          
      if cv2.contourArea(c) ==  max(S):
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box)  if imutils.is_cv2() else cv2.boxPoints( box)
        box = np.array(box, dtype="int")
        box = perspective.order_points( box)
        orig = cv2.drawContours(orig, [box.astype("int")] , -1 , ( 0 , 255 , 0 ) , 2)
        for ( x, y)  in box:
          cv2.circle(orig, (int(x), int(y)) , 5 , ( 0 , 0 , 255 ) , -1)
        ( tl, tr, br, bl) = box
        
        (tltrX, tltrY) = midpoint( tl, tr)
        ( blbrX, blbrY) = midpoint( bl, br)
        ( tlblX, tlblY) = midpoint( tl, bl)
        ( trbrX, trbrY) = midpoint( tr, br)

        ( centerXX, centerYY) = midpoint( tl, br)
        # draw the midpoints on the image
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        cv2.circle(orig, (int(blbrX), int(blbrY)) , 5 , ( 255 , 0 , 0 ) , -1)
        cv2.circle(orig, (int(tlblX), int( tlblY)) , 5 , ( 255 , 0 , 0 ) , -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)) , 5 , ( 255 , 0 , 0 ) , -1)
      # # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        cv2.putText( orig, " {:.1f}in".format(dA),(int(tltrX - 15), int( tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        cv2.putText( orig, " {:.1f}in".format(dB),(int(trbrX + 10), int( trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        return orig

