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
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import split
from tqdm import tqdm

#dcd de ef unzip_file(url: str, target_path: str):
    #response = requests.get(url, stream=True)
    # handle = open(target_path, "wb")
    # for chunk in response.iter_content(chunk_size=512):
    #     if chunk:
    #         handle.write(chunk)
    # handle.close()
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



#extract_json("./dataset/examples")
#save_objects() # total of 30 different pieces

def extract_objects(my_path: str):
    for element in os.listdir(my_path):
        if element.endswith(".jpeg") and element[-6].isnumeric():
            name_image = element.split("/")[-1]
            im = cv2.imread(my_path + "/" + element, cv2.IMREAD_COLOR)
            img_clone = im.copy()
            boxes = bbox_image(name_image)
            polylist = []
            for box in boxes:
                min_x, max_x = boxes[box][0][0], boxes[box][0][0]
                min_y, max_y = boxes[box][0][1], boxes[box][0][1]
                for vertices in boxes[box]:
                    if vertices[0] > max_x:
                        max_x = vertices[0]
                    elif vertices[0] < min_x:
                        min_x = vertices[0]
                    if vertices[1] > max_y:
                        max_y = vertices[1]
                    elif vertices[1] < min_y:
                        min_y = vertices[1]
                box_poly = Polygon([(min_x, min_y), (min_x, max_y),(max_x, min_y), (max_x, max_y)])
                polylist.append(box_poly)
        return polylist
                #rectangle = img_clone[min_y:min_y + (max_y - min_y), min_x:min_x + (max_x - min_x)]
                #cv2.imwrite(name_image + "_" + box + ".jpeg", rectangle)



def intersection_list(polylist: set):
    list_intersected = []
    for element in range(len(polylist)//2):
        r = polylist[element]
        for el in range(len(polylist)//2, len(polylist)):
            p = polylist[el]
            if r.intersects(p):
                list_intersected.append((el, element))
    return list_intersected

def slice_one(gdf, index):
    inter = gdf.loc[gdf.intersects(gdf.iloc[index].geometry)]
    if len(inter) == 1: return inter.geometry.values[0]
    box_A = inter.loc[index].values[0]
    inter = inter.drop(index, axis=0)
    polys = []
    for i in range(len(inter)):
        box_B = inter.iloc[i].values[0]
        polyA, *_ = slice_box(box_A, box_B)
        polys.append(polyA)
    return intersection_list(polys)

def slice_box(box_A:Polygon, box_B:Polygon, margin=10, line_mult=10):
    vec_AB = np.array([box_B.centroid.x - box_A.centroid.x, box_B.centroid.y - box_A.centroid.y])
    vec_ABp = np.array([-(box_B.centroid.y - box_A.centroid.y), box_B.centroid.x - box_A.centroid.x])
    vec_AB_norm = np.linalg.norm(vec_AB)
    split_point = box_A.centroid + vec_AB/2 - (vec_AB/vec_AB_norm)*margin
    line = LineString([split_point-line_mult*vec_ABp, split_point+line_mult*vec_ABp])
    split_box = split(box_A, line)
    if len(split_box) == 1: return split_box, None, line
    is_center = [s.contains(box_A.centroid) for s in split_box]
    where_is_center = np.argwhere(is_center).reshape(-1)[0]
    where_not_center = np.argwhere(~np.array(is_center)).reshape(-1)[0]
    split_box_center = split_box[where_is_center]
    split_box_out = split_box[where_not_center]
    return split_box_center, split_box_out, line




print(intersection_list(extract_objects("./dataset/examples")))



