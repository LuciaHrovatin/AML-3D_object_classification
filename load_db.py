
# pip install matplotlib tqdm torch opencv-python

import os
import os.path
import torch
import glob
import json
import numpy as np
import sys
from tqdm import tqdm #returns a progress bar -> tqdm.tqdm(variable, 'tag')
import matplotlib
import matplotlib.pyplot as plt
#import cv2

def load_db(db_path):
    def load_data(mypath):
        '''
        create an array with all the images.
        In this case it is an array of zeroes. Dimensions chosen as follows:
            0: number of samples (0 = stack them one above the other)
            150: img size (length)
            150: img size (height)
            3: number of channels (RGB) #maybe in our case it should be 4, because we consider RGB-D
        dtype = data type of the destination array
        '''
        data = np.zeros((0, 150, 150, 3), dtype = 'uint8')
        labels = np.zeros((0,))

        for i, cla in enumerate(mypath):
            '''
            i = idx
            cla = class
            '''
            json_files = glob.glob(os.path.join(cla, '*.json'))
            #views = glob.glob(os.path.join(cla, '*_view=*.jpeg'))

            filelist = glob.glob(os.path.join(cla, '*.jpeg'))
            tmp_data = np.empty((len(filelist), 150, 150, 3), dtype='uint8')
            tmp_labels = np.ones((len(filelist),)) * i

            for j,path in enumerate(filelist):
                image = cv2.imread(path) # gets and reads the image
                image = cv2.resize(image, (150, 150))
                tmp_data[j, :] = image
            
            data = np.concatenate((data, tmp_data))
            labels = np.concatenate((labels, tmp_labels))
        
        return data, labels

    
    OG_DIR = os.getcwd()
    ROOT_DIR = os.path.abspath("/Volumes/TOSHIBA_EXT_1")
    db_folder = "examples"
    DB_DIR = glob.glob(os.path.join(ROOT_DIR, db_folder, '*')) # get all the (classes?) documents in the directory
    #sort the classes - do we have classes?
    DB_DIR.sort()
    
    db_data, db_labels = load_data(DB_DIR)
    
    return db_data, db_labels


###################################################################################
###################################################################################
###################################################################################

files = os.listdir(DB_DIR)
#print(files)
all_json_names = [file for file in files if file.endswith(".json")] # naïve solution - change it asap!
                                                                    # plus - they are the filenames as STRINGS, NOT THE FILES!!

#print(all_json_names, len(all_json_names))
# LAST JSON: scene9-30_view=9.json, len = 1800

#print(os.path.abspath("/Volumes/TOSHIBA_EXT_1/examples/{}".format(all_json_names[0])))

### STEP 0 (MAYBE NOT NEEDED): move all the files in a BIG json file to have everything on hand

############ NEED TO WORK ON THIS FILE

'''os.chdir(DB_DIR)
d = {}
for file in all_json_names:
    with open(file,"r") as f:
        for key in f:
            if key not in d:
                d[key] = []
            d[key].append(f[key])    #TypeError: _io.TextIOWrapper' object is not subscriptable


with open("all_objects.json","w") as f:
    json.dump(f,d)'''

### STEP 1: Image segmentation
'''
source: https://data-flair.training/blogs/image-segmentation-machine-learning/

We are going to perform image segmentation using the Mask R-CNN architecture,
which returns the binary object mask in addition to class label
and object bounding box. Mask R-CNN is good at pixel level segmentation.
'''

"""img = cv2.imread('scene0-20_view=0.jpeg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,8))
plt.imshow(img,cmap="gray")
plt.axis('off')
plt.title("Original Image")
plt.show()"""