
##extract single images from kauri data
##4 Aug 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import load_img

# Helper libraries
import numpy as np
from matplotlib import pyplot
import matplotlib

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image

from tensorflow.python.keras import backend as K
from PIL import Image
import math
import os

import lxml.etree as ET
import efficientnet.tfkeras as efn
import time

##imageloc = r'C:\DeepLearning\COMPX591\Data\images\test'
imageloc = '/Scratch/dans/kauri/images/test'
##annotation = r'D:\Study\COMPX591\Data\images\test\20190409-WEM_6166.xml'

##model = r'C:\DeepLearning\COMPX591\newmodels\model_efficientnet.h5'
model = '/home/db129/kauri/scripts/model_efficientnet.h5'

##ouaut = r'C:\DeepLearning\COMPX591\Data\Crawler\Level2_270x180' 
ouaut = '/home/db129/kauri/Results'
##img = load_img(imageloc)

cropwidth = 80  
cropheight = 160
vertstep = 30
horizontalstep = 30

loaded_model = tf.keras.models.load_model(model)
loaded_model.layers[0].input_shape #(None, 160, 160, 3)

def mergeoverlap(allareas):
    print ('merging areas')
    intersects = []
    areas = allareas
    newareas = []
    intersectsnew = []
    for a in range(len(areas)):
        areaA = areas[a]
        intersection = []
        axmin = areaA[0]
        aymin = areaA[1]
        axmax = areaA[2]
        aymax = areaA[3]
        for b in range(len(areas)):
            areaB = areas[b]
            bxmin = areaB[0]
            bymin = areaB[1]
            bxmax = areaB[2]
            bymax = areaB[3]
            x_overlap = max(0, min(axmax, bxmax) - max(axmin, bxmin))
            y_overlap = max(0, min(aymax, bymax) - max(aymin, bymin))
            overlap =  x_overlap * y_overlap
            areaA_area = (axmax - axmin) * (aymax - aymin)
            areaB_area = (bxmax - bxmin) * (bymax - bymin)
            overlap_perc = max(overlap/areaA_area, overlap/areaB_area)
            if overlap_perc > 0.2:
                intersection.append(b)
        intersects.append(intersection)

    for intersecta in intersects:
        ##merge = intersecta
        intersectaset = set(intersecta)
        superset = []
        for intersectb in intersects:
            intersectbset = set(intersectb)
            if len(intersectaset.intersection(intersectbset)) >0:
                 intersectaset =  intersectaset.union(intersectbset)
        intersetlist = list(intersectaset)
        intersectsnew.append(intersetlist)

    intersects_reduced = []
    for elem in intersectsnew:
        if elem not in intersects_reduced:
            intersects_reduced.append(elem)
    intersectsnew = intersects_reduced

    for intersect in intersectsnew:
        i=0
        for areaindex in intersect:
            area = areas[areaindex]
            if i==0:
                cxmin = area[0]
                cymin = area[1]
                cxmax = area[2]
                cymax = area[3]
            else:
                cxmin = min(cxmin,area[0])
                cymin = min(cymin,area[1])
                cxmax = max(cxmax,area[2])
                cymax = max(cymax,area[3])
            i=i+1
        area = (cxmin, cymin, cxmax, cymax )
        newareas.extend([area])
       
    return newareas


def converttoxml(areas, ouaut, w, ht):
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder')
    ET.SubElement(root, 'filename')
    ET.SubElement(root, 'path')
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(w)
    ET.SubElement(size, 'height').text = str(ht)
    ET.SubElement(size, 'depth').text = '3'
    ET.SubElement(root, 'segmented').text= '0'
    for area in areas:
        xmin = str(area[0])
        ymin = str(area[1])
        xmax = str(area[2])
        ymax = str(area[3])
        object = ET.SubElement(root, 'object')
        ET.SubElement(object, 'name').text = 'emergent'
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = xmin
        ET.SubElement(bndbox, 'ymin').text = ymin
        ET.SubElement(bndbox, 'xmax').text = xmax
        ET.SubElement(bndbox, 'ymax').text = ymax   

    tree_out = ET.tostring(root, pretty_print=True)
    with open(ouaut, 'wb') as f:
        f.write(tree_out)

def getdiffareas(area):
    newareas = []
    xmin = area[0]
    ymin = area[1]
    xmax = area[2]
    ymax = area[3]
    xcentre = xmax-cropwidth/2
    ycentre = ymax-cropheight/2
    xfactor = 80

    while xfactor < 200:
        newxmin = int(max(xcentre - xfactor/2,0))
        newxmax = int(xcentre + xfactor/2)
        yfactor = 160
        while yfactor < 300:
            newymin = int(max(ycentre - yfactor/2,0))
            newymax = int(ycentre + yfactor/2)
            newarea  = (newxmin, newymin, newxmax, newymax)
            newareas.extend([newarea])
            yfactor = yfactor + 30
        xfactor = xfactor + 20

    return newareas

start_time = time.time()
for file in os.listdir(imageloc):

    h=0
    if file.endswith('6166.jpg'):
        img = load_img(imageloc + '/' + file)
        annotation = imageloc + '/' + file[:-4] + '.xml'
        imagewidth = img.size[0]
        imageheight = img.size[1] - 285
        
        print(imagewidth)
        print(img.size)
        
        areas = []
        xmin =0
        ymin =0
        for h in range(round((imageheight-cropheight)/(vertstep))):
            if h % 10==0:
                print ('iteration is ' + str(h))
                print ('no of areas is ' + str(len(areas)))
            if h % 2 ==0:
                xmin =0
            else:
                xmin = 15
            ##for i in range(): ##round((imagewidth-cropwidth-xmin)/horizontalstep)):
            while xmin < imagewidth-cropwidth:
                xmax = xmin + cropwidth
                ymax = ymin + cropheight
                area = (xmin, ymin, xmax, ymax )
                cropped = img.crop(area)
                cropped_img = cropped.resize((224,224))
                cropped_img = np.expand_dims(cropped_img, axis=0)
                ##result=loaded_model.predict_classes(cropped_img)
                pred = loaded_model.predict(cropped_img)
                ##if pred[0][0] > 9.5251075e-04:
                ##print ("xmin: " + str(xmin) + " ,ymin: " + str(ymin) + " " + str(pred))
                predflt = float(pred[0][0])
                if predflt > 0.8:##result[0] ==0:
                    ##result2=loaded_model.predict_classes(cropped_img)
                    ##pred2 = loaded_model.predict(cropped_img)
                    ##print ("pred2 " + str(pred2))
                    balloonareas = getdiffareas(area)
                    maxpred = 0
                    areatoadd = area
                    predcount = 0
                    for balloonarea in balloonareas:
                        cropped2 = img.crop(balloonarea)
                        cropped_img2 = cropped2.resize((224,224))
                        cropped_img2 = np.expand_dims(cropped_img2, axis=0)
                        ##result2=loaded_model2.predict_classes(cropped_img2)
                        pred2 = loaded_model.predict(cropped_img2)
                        pred2flt = float(pred2[0][0])
                        print (str(balloonarea) + ', ' + str(pred2[0][0]))
                        if pred2flt > 0.9:
                            predcount = predcount +1
                        if pred2flt > maxpred:
                            maxpred = pred2flt
                            areatoadd = balloonarea
                    if predcount >17:
                        print ('pred is ' + str(maxpred) + ', predcount ' + str(predcount))
                        print ("adding area" + str(areatoadd)) 
                        areas.extend([areatoadd])
                        ##xmin = xmax - horizontalstep ## move along size of area added
                    #if result2[0] ==0:
                    #    areas.extend([area])
                    #    print ("adding area")

                xmin=xmin+horizontalstep
            ymin = ymin + vertstep
        converttoxml(areas, ouaut  +'/' +  file[:-4] + '_areasall.xml', img.size[0], img.size[1])
        areas = mergeoverlap(areas)
        converttoxml(areas, ouaut  +'/' +  file[:-4] + '_areas.xml', img.size[0], img.size[1])
print("--- %s seconds ---" % (time.time() - start_time))

            





