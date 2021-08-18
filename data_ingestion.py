import json
import requests
import os
from zipfile import ZipFile
import numpy as np
import cv2

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
        if element.endswith("[0-9]+.jpeg"):
            print("Done")
            name_image = element.split("/")[-1]
            im = cv2.imread(name_image)
            boxes = bbox_image
            for box in boxes:
                print("Done")
                for vertices in box[0]:
                    min_x,max_x = 0, 0
                    min_y, max_y = 0, 0
                    if vertices[0] >= max_x:
                        max_x = vertices[0]
                    elif vertices[0] <= min_x:
                        min_x = vertices[0]
                    if vertices[1] >= max_y:
                        max_y = vertices[1]
                    elif vertices[1] <= min_y:
                        min_y = vertices[1]
                    low_sx = min_x, min_y # lower bound left
                    low_dx = max_x, min_y # lower bound right
                    up_sx = min_x, max_y  # upper bound left
                    up_dx = max_x, max_y  # upper bound right
                rectangle = im[min_y:min_y + abs(max_y - min_y), min_x:min_x + abs(max_x - min_x)]
                cv2.imshow(rectangle)
                print("Done")
                break

extract_objects("./dataset/examples")





