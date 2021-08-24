import json
import os
from zipfile import ZipFile
import numpy as np
import cv2
from PIL import Image
#import open3d as o3d


class DataIngestion:

    def __init__(self):
        """
        The class DataIngestion is initialised specifying the name
        of the json file where the transformed dataset will be stored.
        """
        self.data_storer = "final_db.json"
        self.path = "dataset"

    def unzip_file(self, zip_folder: str):
        """
        Extracts all the files stored in the zip file saving
        them in the current directory in the "dataset" folder.
        @param zip_folder: name of the zipped folder from which the files are extracted
        """
        with ZipFile(zip_folder, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path=self.path)

    def extract_json(self, my_path: str):
        """
        This function detects and parses the json files in the folder "examples".
        @param my_path: directory of the folder
        @return final_db: dictionary with the following structure {image_name: {y : [bbox]}}
        """
        final_db = dict()
        for element in os.listdir(my_path):
            if element.endswith(".json"):
                name_image = element.rstrip(".json").split("/")[-1]
                if name_image not in final_db:
                    storing = self.store_info(my_path + "/" + element)
                    final_db[name_image] = storing
        stringJSON = json.dumps(final_db)
        with open(self.data_storer, "w") as f:
            f.write(stringJSON)

    @staticmethod
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

    def get_objects(self) -> set:
        """
        MODIFICA per restituire un int (N classi)
        Saves the number of distinct classes (i.e., lego pieces) and returns them
        """
        with open(self.data_storer, ) as f:
            data = json.load(f)
            legos = set()
            for scene in data:
                for key in data[scene]:
                    legos.add(key)
            return legos

    def bbox_image(self, image_name: str):
        """
        The function takes as input the name of an image and
        returns the list of bounding boxes stored in the json file.
        If the name is not found, the function yields an error.
        @param image_name: name of the image
        @return the list of bounding boxes
        """
        image = image_name.rstrip(".jpeg")
        with open(self.data_storer, ) as f:
            data = json.load(f)
            try:
                return data[image]
            except ValueError:
                return print("Scene " + image + " is not present in the dataset.")

    def extract_objects(self, my_path: str):
        tot_images = os.listdir(my_path)
        #total = 0
        #actual = 0
        for element in tot_images:
            if element.endswith(".jpeg") and element[-6].isnumeric():
                name_image = element.split("/")[-1]
                boxes = self.bbox_image(name_image)
                for box in boxes:
                    #actual += len(set(tuple(x) for x in boxes[box]))
                    #total += len(boxes[box])
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

                        for vertices in boxes[box]:
                            if 0 < vertices[0] < 1023:
                                if vertices[0] > max_x:
                                    max_x = vertices[0]
                                elif vertices[0] < min_x:
                                    min_x = vertices[0]
                            if 0 < vertices[1] < 1023:
                                if vertices[1] > max_y:
                                    max_y = vertices[1]
                                elif vertices[1] < min_y:
                                    min_y = vertices[1]
                        if min_y != max_y and min_x != max_x:
                            im = cv2.imread(my_path + element, cv2.IMREAD_COLOR)
                            im_depth = cv2.imread(my_path + name_image + "depth.jpeg", cv2.IMREAD_GRAYSCALE)
                            rectangle = im[min_y: min_y + (max_y - min_y), min_x: min_x + (max_x - min_x)]
                            rectangle_depth = im_depth[min_y: min_y + (max_y - min_y), min_x: min_x + (max_x - min_x)]
                            savedPath = os.getcwd()
                            dir = "images_final"
                            if not os.path.exists(dir):
                                os.mkdir(dir)
                            os.chdir(dir)
                            #cv2.imwrite(name_image + "_" + box + ".jpeg", rectangle)
                            #cv2.imwrite(name_image + "_" + box + "_depth.jpeg", rectangle_depth)
                            cv2.imwrite(name_image  + "_" + box + ".jpeg", point_cloud(rectangle, rectangle_depth))
                            os.chdir(savedPath)
        #return (actual/total) * 100

    # in a z-map every pixel in a scene is assigned a 0-255 grayscale value based upon its distance from the camera.
    # Traditionally the objects
    # - closest to the camera are white
    # - the objects furthest from the camera are black
    # A depth map only contains the distance or Z information for each pixel
    # which in a monochrome (grayscale) 8-bit representation is necessary with values between [0, 255],
    # where 255 represents the closest possible depth value and 0 the most distant possible depth value.

    @staticmethod
    def point_cloud(image_col, image_depth):
        """
        Giving two frames, a colored and a grey one, of the same lego piece,
        the function returns the corresponding Point Cloud.
        @param image_col: colored frame
        #param image_depth: image reporting depth information
        """
        color_raw = o3d.io.read_image(image_col)
        depth_raw = o3d.io.read_image(image_depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        #print(rgbd_image)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        return pcd
        # Flip it, otherwise the pointcloud will be upside down
        #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #o3d.visualization.draw_geometries([pcd], zoom=0.5)


    @staticmethod
    def transform_binary(point_cloud: str):
        # Read Image
        img = Image.open(point_cloud)
        # Convert Image to Numpy as array
        img = np.array(img)
        # Put threshold to make it binary
        binarr = np.where(img > 128, 255, 0)
        # Covert numpy array back to image
        binimg = Image.fromarray(binarr)


example = DataIngestion()
print(example.extract_objects("./dataset/examples/"))

"""
# NON NECESSARIA!!
def intersection_list(polylist: set):
    list_intersected = []
    for element in range(len(polylist) // 2):
        r = polylist[element]
        for el in range(len(polylist) // 2, len(polylist)):
            p = polylist[el]
            if r.intersects(p):
                iou = r.intersection(p).area / r.union(p).area
                list_intersected.append([el, element, iou])

    return list_intersected
"""