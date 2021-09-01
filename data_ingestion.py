import json
import os
from zipfile import ZipFile
import numpy as np
import cv2
import csv
import open3d as o3d


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

    def get_objects(self) -> int:
        """
        Saves distinct classes (i.e., lego pieces) and
        returns the final number of classes.
        @return integer number of classes
        """
        with open(self.data_storer, ) as f:
            data = json.load(f)
            legos = set()
            for scene in data:
                for key in data[scene]:
                    legos.add(key)
            return len(legos)

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
        total = 0
        actual = 0
        for element in tot_images:
            if element.endswith(".jpeg") and element[-6].isnumeric():
                name_image = element.split("/")[-1]
                boxes = self.bbox_image(name_image)
                for box in boxes:
                    actual += len(set(tuple(x) for x in boxes[box]))
                    total += len(boxes[box])
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
                            im_depth = cv2.imread(my_path + name_image.rstrip(".jpeg") + "_depth.jpeg",
                                                  cv2.IMREAD_GRAYSCALE)
                            if im_depth is not None and im is not None:
                                rectangle = im[min_y: min_y + (max_y - min_y), min_x: min_x + (max_x - min_x)]
                                rectangle_depth = im_depth[min_y: min_y + (max_y - min_y), min_x: min_x + (max_x - min_x)]
                                savedPath = os.getcwd()
                                new_dir = "images_final"
                                if not os.path.exists(new_dir):
                                    os.mkdir(new_dir)
                                os.chdir(new_dir)
                                cv2.imwrite(name_image.rstrip(".jpeg") + "_" + box + ".jpeg", rectangle)
                                cv2.imwrite(name_image.rstrip(".jpeg") + "_" + box + "_depth.jpeg", rectangle_depth)
                                os.chdir(savedPath)
        return print((actual / total) * 100)

    @staticmethod
    def point_cloud(image_col, image_depth):
        """
        Giving two frames, a colored one (RGB image) and a monochromatic one (depth map), of the same lego block,
        the function returns its name (unique identifier) and Point Cloud.
        @param image_col: colored frame
        @param image_depth: image reporting depth information where each pixel is assigned a 0-255 grayscale value,
        where 255 represents the closest depth value and 0 the most distant one.
        @return a list having as first element the unique identifier of the lego block
        and as second element the Point Cloud as list of arrays.
        """
        color_raw = o3d.io.read_image(image_col)
        depth_raw = o3d.io.read_image(image_depth)

        # create a RGB-D image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

        # create a point cloud (upside down)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # flip the point cloud
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lego_block = image_col.split("/")[-1].split("_")[-1].strip(".jpeg")
        return [lego_block, np.asarray(pcd.points)]

    def transform_csv(self, my_path: str):
        """
        Saves a csv file of each Lego block and its point cloud representation.
        @param my_path: string containing the path to the folder storing the frames
        """
        tot_images = os.listdir(my_path)
        with open("final_images.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["lego_name", "point_cloud"])
            writer.writeheader()

            for el in tot_images:
                # exclude depth maps
                if "depth" not in el:
                    # RGB image
                    im_col = my_path + "/" + el

                    # corresponding depth map
                    im_depth = my_path + "/" + el.strip(".jpeg") + "_depth.jpeg"

                    data = self.point_cloud(image_col=im_col, image_depth=im_depth)
                    writer.writerow({'lego_name': data[0], 'point_cloud': data[1]})
