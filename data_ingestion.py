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
#import open3d as o3d

def unzip_file(zipfolder: str):
    """
    Extracts all the files stored in the zip file saving
    them in the current directory in the "dataset" folder.
    @param filename: name of the zipped folder from which the files are extracted
    """
    with ZipFile(zipfolder, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
        zipObj.extractall(path='dataset')


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


def store_info(element: str) -> dict:
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

def save_objects() -> set:
    """
    Saves the number of distinct classes (i.e., lego pieces) and returns them
    """
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
        for box in boxes:
            if len(set(tuple(x) for x in boxes[box])) == len(boxes[box]):
                if boxes[box][0][0] < 0 or boxes[box][0][0] > 1023:
                    if boxes[box][0][0] < 0:
                        boxes[box][0][0] = 0
                    else:
                        boxes[box][0][0] = 1023
                if boxes[box][0][1] < 0 or boxes[box][0][1] > 1023:
                    if boxes[box][0][1] < 0:
                        boxes[box][0][1] = 0
                    else:
                        boxes[box][0][1] = 1023
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
                if min_y != max_y and min_x != max_x:
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








extract_objects("./dataset/examples/scene0-20_view=1.jpeg")


# in a z-map every pixel in a scene is assigned a 0-255 grayscale value based upon its distance from the camera.
# Traditionally the objects
# - closest to the camera are white
# - the objects furthest from the camera are black
# A depth map only contains the distance or Z information for each pixel
# which in a monochrome (grayscale) 8-bit representation is necessary with values between [0, 255],
# where 255 represents the closest possible depth value and 0 the most distant possible depth value.
"""
def point_cloud(image_col, image_depth):
    color_raw = o3d.io.read_image(image_col)
    depth_raw = o3d.io.read_image(image_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    print(rgbd_image)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd], zoom=0.5)
"""
