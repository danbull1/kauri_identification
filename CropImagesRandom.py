
##extract single images from kauri data
##4 Aug 2019


import csv
from PIL import Image
import random


rcsvfile = r'D:\Study\COMPX591\Data\images\train.csv'
imageloc = r'D:\Study\COMPX591\Data\images\train'
saveloc = r'D:\Study\COMPX591\Data\singleimages2\train\random'

def checkforintersect(width, height, cropwidth, cropheight, imagename):
    print ('checking for intersect')
    isintersect = False
    xmin = width
    ymin = height
    xmax = width + cropwidth
    ymax = height + cropheight
    
    with open(rcsvfile) as csv_filecheck:
        
        filecheck = csv.reader(csv_filecheck, delimiter=',')
        for row in filecheck:
            imagetocheck = row[0]
            if imagetocheck == imagename:
                if row[0] !='filename':      
                    chkxmin = int(row[4]) - 10
                    chkymin = int(row[5])
                    chkxmax = int(row[6]) + 10
                    chkymax = int(row[7])

                    x_overlap = max(0, min(xmax, chkxmax) - max(xmin, chkxmin))
                    y_overlap = max(0, min(ymax, chkymax) - max(ymin, chkymin))
                    overlap =  x_overlap * y_overlap
                    if overlap > 0:
                        isintersect = True
                        print ('there is an intersect ')
    return isintersect




with open(rcsvfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            imagename = row[0]
            imagelabel = row[3]
            img = Image.open(imageloc + '\\' + imagename)
            width, height = img.size
            height = height - 384 ##allow for grey at bottom of image
            cropwidth = int(row[6]) - int(row[4])
            cropheight = int(row[7]) - int(row[5])
            
            rndwidth = random.randint(1,width -cropwidth)
            rndheight = random.randint(1, height-cropheight)

            hasintersect = checkforintersect(rndwidth, rndheight, cropwidth, cropheight, imagename)

             ##pick radnom again
            while hasintersect:
                rndwidth = random.randint(1,width -cropwidth)
                rndheight = random.randint(1, height-cropheight)
                hasintersect = checkforintersect(rndwidth, rndheight, cropwidth, cropheight, imagename)

            if hasintersect==False:
                area = (rndwidth, rndheight, rndwidth + cropwidth, rndheight + cropheight)
                print (area)
                cropped_img = img.crop(area)
                cropped_img.save(saveloc + '\\random_' + imagename[:-4] + '_W' + str(rndwidth) + '_H' + str(rndheight)+'.jpg')
                line_count += 1

