# Applied Machine Learning Project
## 3D Object Classification with PointNet

The **3D Object Classification with PointNet** project has been developed as final assignment of the Applied Machine Learning course, offered by the University of Trento.

## Project objective 
The project aims at deploying an architecture able to perform 3D Lego blocks extraction, transformation and classification by means of [PointNet](https://arxiv.org/abs/1612.00593) approach. Specifically, the employed dataset has been cleaned and stored in a zip folder called [images_final.zip](https://drive.google.com/file/d/10B4uLcfnGG-srzVUV8F2Lq3v_a9kPoz1/view?usp=sharing).  

**Note:** the dataset can be accessed and downloaded only with a *unitn* account.  

## Prerequisites 

In order to run this project, the following tools have to be installed on your machine: 
- Python, preferably [3.8](https://www.python.org/downloads/release/python-380/) 

**Note:** Python version [3.9](https://www.python.org/downloads/release/python-390/) may not be supported by some modules (for futher information and updates check [open3d](http://www.open3d.org/docs/latest/getting_started.html)).   

## Installation 

### Clone the repository 

Clone this repository in a local directory typing in the command line: 

```
git clone https://github.com/LuciaHrovatin/AML-3D_object_classification.git
```

### Environment 
The creation of a virtual environment is highly suggested. If not already installed, install virtualenv:

- in Unix systems:
    ```
    python3 -m pip install --user virtualenv
    ```

- in Windows systems:
    ```
    python -m pip install --user virtualenv
    ```

And then create the virtual environment named *venv* typing in the command line (inside the project folder): 

- in Unix systems:
    ```
    python3 -m venv venv
    ```

- in Windows systems:
    ```
    python -m venv venv
    ```

The virtual environment can be activated as follow: 

- in Unix systems:
    ```
    source venv
    ```

- in Windows systems:
    ```
    venv\Scripts\activate
    ```
### Requirements 

In the active virtual environment, install all libraries contained in the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage 
The project can be run starting from different datasets stored in zipped folders. 

### Option 1: "lego_dataset.zip"  
Download the [lego_dataset.zip](https://drive.google.com/file/d/1fohALgsFKM8VXY1pxoBkeANy_f52IdtL/view?usp=sharing) and save it in the project directory. Uncomment the lines specified in the `main.py` file.   

### Option 2: "images_final.zip"  
Download the cleaned dataset from [images_final.zip](https://drive.google.com/file/d/10B4uLcfnGG-srzVUV8F2Lq3v_a9kPoz1/view?usp=sharing) and save it in the project directory. Uncomment the lines specified in the `main.py` file.   

### Run the script 
After the virtual environment and the zip folder set up (either option 1 or 2), a last step must be performed. To start the entire architecture, type in the command line (with the activated virtual environment): 

- in Unix systems:
    ```
    python3 main.py
    ```

- in Windows systems:
    ```
    python main.py
    ```

# Code structure

The backend code structure is composed by:
-   `data_ingestion.py`, containing the functions used to the cleansing procedure applied on the original dataset 
-   `data_splitter.py`, containing the functions for splitting the dataset in training and validation set 
-   `model.py`, contains the PointNet architecture  
-   `solver.py`, launches the training/testing procedure emplying Adam algorithm, Cross-Entropy loss function and Accuracy   
-   `main.py`, triggers the entire project 
-   `final_db.json`, stores the scenes and the bounding boxes coordinates for each Lego block  

## Overall code structure
```
├── .gitignore
├── data_ingestion.py
├── data_splitter.py
├── final_db.json
├── model.py
├── solver.py
├── requirements.txt
└── main.py
```
