import json
import re
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
from zipfile import ZipFile
import numpy as np
import cv2
import shapely
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import split
from tqdm import tqdm

#def unzip_file(url: str, target_path: str):
    #response = requests.get(url, stream=True)
    # handle = open(target_path, "wb")
    # for chunk in response.iter_content(chunk_size=512):
    #     if chunk:
    #         handle.write(chunk)
    # handle.close()
# to unzip the dataset
#with ZipFile("lego_dataset.zip", 'r') as zipObj:
    # Extract all the contents of zip file in current directory
#    zipObj.extractall(path='dataset')


def extract_json(my_path):
    """
    This function detects and parses the json files in the folder "examples".
    @param mypath: directory of the folder
    @return final_db: dictionary with the following structure {image_name: {y : [bbox]}}
    """
    final_db = dict()
    for element in os.listdir(my_path):
        if element.endswith(".json"):
            name_image = element.rstrip(".json").split("/")[-1]
            if name_image not in final_db:
                storing = store_info(my_path + "/" + element)
                final_db[name_image] = storing
    stringJSON = json.dumps(final_db)
    with open("final_db.json", "w") as f:
        f.write(stringJSON)


def store_info(element: str):
    """
    It parses the json file passed as parameter, returning a dictionary
    with y (lego names) as keys and bbox as values.
    @param element: name of the json
    @return scene: dictionary with structure {y: [bbox]}
    """
    scene = dict()
    with open(element, "r") as f:
        data = json.load(f)
        for obj in data:
            for key in data[obj]:
                if key == "y":
                    identifier = data[obj][key]
                if key == "bbox":
                    bbox = data[obj][key]
            scene[identifier] = bbox
    return scene

def save_objects():
    with open("./final_db.json",) as f:
        data = json.load(f)
        legos = set()
        for scene in data:
            for key in data[scene]:
                legos.add(key)
        return legos

def bbox_image(filename: str):
    image = filename.rstrip(".jpeg")
    with open("./final_db.json", ) as f:
        data = json.load(f)
        try:
            return data[image]
        except ValueError:
            return print("Scene" + image + "is not present in the dataset.")


def extract_objects(my_path: str):
    element = my_path
    #for element in os.listdir(my_path):
    if element.endswith(".jpeg") and element[-6].isnumeric():
        name_image = element.split("/")[-1]
        boxes = bbox_image(name_image)
        polylist = []
        for box in boxes:
            if boxes[box][0][0] < 0 or boxes[box][0][0] > 1023:
                if boxes[box][0][0] < 0:
                    boxes[box][0][0] = 0
                else:
                    boxes[box][0][0] = 1022
            if boxes[box][0][1] < 0 or boxes[box][0][1] > 1023:
                if boxes[box][0][1] < 0:
                    boxes[box][0][1] = 0
                else:
                    boxes[box][0][1] = 1022
            min_x, max_x = boxes[box][0][0], boxes[box][0][0]
            min_y, max_y = boxes[box][0][1], boxes[box][0][1]

            print("these:", max_y, min_y, min_x, max_x)
            for vertices in boxes[box]:
                if vertices[0] >= max_x and vertices[0] > 0 and vertices[0] < 1023:
                    max_x = vertices[0]
                elif vertices[0] < min_x and vertices[0] > 0 and vertices[0] < 1023:
                    min_x = vertices[0]
                if vertices[1] >= max_y and vertices[1] > 0 and vertices[1] < 1023:
                    max_y = vertices[1]
                elif vertices[1] < min_y and vertices[1] > 0 and vertices[0] < 1023:
                    min_y = vertices[1]
            print("these:", min_y, min_y + (max_y - min_y), "and", min_x, min_x + (max_x - min_x))
            im = cv2.imread(my_path , cv2.IMREAD_COLOR)
            rectangle = im[min_y: min_y + (max_y - min_y), min_x : min_x + (max_x - min_x)]

            cv2.imwrite(name_image + "_" + box + ".jpeg", rectangle)
            #box_poly = shapely.geometry.box(min_x, min_y, max_x, max_y)
                #polylist.append(box_poly)
        #return polylist
                #rectangle = img_clone[min_y:min_y + (max_y - min_y), min_x:min_x + (max_x - min_x)]
                #cv2.imwrite(name_image + "_" + box + ".jpeg", rectangle)



def intersection_list(polylist: set):
    list_intersected = []
    for element in range(len(polylist)//2):
        r = polylist[element]
        for el in range(len(polylist)//2, len(polylist)):
            p = polylist[el]
            if r.intersects(p):
                iou = r.intersection(p).area/r.union(p).area
                list_intersected.append([el, element, iou])

    return list_intersected








extract_objects("./dataset/examples/scene0-20_view=0.jpeg")



