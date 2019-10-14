
##extract single images from kauri data
##4 Aug 2019


import csv
from PIL import Image

#rcsvfile = r'D:\Study\COMPX591\Data\images\verticalshaped2.csv'
#imageloc = r'D:\Study\COMPX591\Data\images\verticalshaped2'
#saveloc = r'D:\Study\COMPX591\Data\singleimages\train'

rcsvfile = r'D:\Study\COMPX591\Data\images\train2\train2.csv'
imageloc = r'D:\Study\COMPX591\Data\images\train2'
saveloc = r'D:\Study\COMPX591\Data\singleimages2\new'

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
            area = (int(row[4]), int(row[5]), int(row[6]), int(row[7]))
            print (area)
            cropped_img = img.crop(area)
            cropped_img.save(saveloc + '\\' + imagelabel + '_' + imagename[:-4] + '_' + str(line_count) +'.jpg')
            line_count += 1