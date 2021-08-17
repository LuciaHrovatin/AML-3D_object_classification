import json
import requests
import os
from zipfile import ZipFile

#def unzip_file(url: str, target_path: str):
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
                final_db[name_image] = store_info(my_path + "/" + element)
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

extract_json("./dataset/examples")







