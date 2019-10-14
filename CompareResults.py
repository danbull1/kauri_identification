import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import csv
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

##generatedannotation = r'C:\DeepLearning\COMPX591\Data\FinalResults\Final#1'
##experannotation = r'C:\DeepLearning\COMPX591\Data\images\test\20190409-WEM_6166.xml'
imageloc = r'C:\DeepLearning\COMPX591\Data\images\test'
saveloc = r'C:\temp\test'


##load data in to pandas for use in analysis
def getdata(xml):
    xml_list = []
    tree = ET.parse(xml)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
        xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

##look at two boxes, do they overlap and by how much
def getoverlap(genarea, exparea):
    overlapscore = 0
    genxmin = genarea[0]
    genymin = genarea[1]
    genxmax = genarea[2]
    genymax = genarea[3]
       
    expxmin = exparea[0]
    expymin = exparea[1]
    expxmax = exparea[2]
    expymax = exparea[3]
    x_overlap = max(0, min(genxmax, expxmax) - max(genxmin, expxmin))
    y_overlap = max(0, min(genymax, expymax) - max(genymin, expymin))
    overlap =  x_overlap * y_overlap
    genarea_area = (genxmax - genxmin) * (genymax - genymin)
    exparea_area = (expxmax - expxmin) * (expymax - expymin)
    ##union_area = genarea_area + exparea_area - overlap
    perc_overlap = overlap/exparea_area *100

    if perc_overlap > 50:
        overlapscore = 2
    elif perc_overlap > 0:
        overlapscore = 1         
    return overlapscore

#gen = getdata(generatedannotation)
#exp = getdata(experannotation)

##return image with predicted and expert annotations
def drawimge(jpg, filename, generatedareas, expertareas):
    image = Image.open(jpg)
    width, height = image.size
    newwid = int(width/10)
    newheight = int(height/10)
    imagesmall = image.resize((newwid, newheight), Image.ANTIALIAS)
    draw = ImageDraw.Draw(imagesmall)
    for area in generatedareas:
        xmin = area[0]/10
        ymin = area[1]/10
        xmax = area[2]/10
        ymax = area[3]/10
        draw.rectangle(((xmin, ymin), (xmax, ymax)), width=3, outline='yellow')
        quality_val = 80
    for area in expertareas:
        xmin = area[0]/10
        ymin = area[1]/10
        xmax = area[2]/10
        ymax = area[3]/10
        draw.rectangle(((xmin, ymin), (xmax, ymax)), width=3, outline='red')
        quality_val = 80

    imagesmall.save(filename + '.jpg', 'JPEG', quality=quality_val)

##iterate through data and work out if expert data and predicted data overlap and by how much
for file in os.listdir(imageloc):
    h=0
    if file.endswith('6166.jpg'):
        generatedannotation =  saveloc + '\\' + file[:-4] + '_areas.xml' 
        experannotation =  imageloc + '\\' + file[:-4] + '.xml'
        gen = getdata(generatedannotation)
        exp = getdata(experannotation)
        expareas = []
        genareas = []
        genareasstr = []
        for i, j in exp.iterrows():
            ##if j[3] == 'emergent':
            xmin = int(j[4])
            ymin = int(j[5])
            xmax = int(j[6])
            ymax = int(j[7])
            area = (xmin,ymin,xmax,ymax)
            expareas.extend([area])

        for i, j in gen.iterrows():
            xmin = int(j[4])
            ymin = int(j[5])
            xmax = int(j[6])
            ymax = int(j[7])
            area = (xmin,ymin,xmax,ymax)
            genareas.extend([area])

        drawimge(imageloc + '\\' + file, saveloc + '\\' + file[:-4], genareas,expareas)

        ## get true positives - i.e. expert data and predicted data agree by more than 50%
        ## get false positives
        maxoverlaps = []
        for genarea in genareas:
            overlap = 0
            overlaps = []
            for exparea in expareas:
                overlap = getoverlap(genarea, exparea)
                overlaps.append(overlap)
            maxoverlap = max(overlaps)
            result = 'fp'
            if maxoverlap == 2:
                result = 'tp'
            if maxoverlap == 1:
                result = 'sp'
            genareasstr.append(['model generateed data', file, str(genarea[0]),str(genarea[1]), str(genarea[2]),str(genarea[3]), result])

        ## get false negatives
        maxoverlaps2 = []
        for exparea in expareas:
            overlap = 0
            overlaps = []
            for genarea in genareas:
                overlap = getoverlap(exparea, genarea)
                overlaps.append(overlap)
            maxoverlap2 = max(overlaps)
            result = 'fn'
            if maxoverlap2 == 2:
                result = 'tp'
            if maxoverlap2 == 1:
                result = 'sp'
            print (exparea)
            print (result)
            if result == 'fn':
                genareasstr.append(['expert annotated data', file, str(genarea[0]),str(genarea[1]), str(genarea[2]),str(genarea[3]), result])

        with open(saveloc + '\\results_all2' +'.csv', mode='a', newline='') as results:
            results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for g in genareasstr:
                results_writer.writerow(g)

