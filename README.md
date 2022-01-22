# Applied Machine Learning Project
## 3D Object Classification with PointNet

The **3D Object Classification with PointNet** project has been developed as final assignment of the Applied Machine Learning course, offered by the University of Trento during the Academic Year 2020/2021.

## Project objective 
The project aims to deploy an architecture able to perform 3D Lego blocks extraction, transformation and classification by means of the [PointNet](https://arxiv.org/abs/1612.00593) architecture. Specifically, the employed dataset has been cleaned and stored in a zip folder called [images_final.zip](https://drive.google.com/file/d/10B4uLcfnGG-srzVUV8F2Lq3v_a9kPoz1/view?usp=sharing) or, alternately, in pickle files: [images_final.pkl](https://drive.google.com/file/d/1CdL_l_6IUfLe5UmuaQ9wPBmuMFW91Xo4/view) and [labels_final.pkl](https://drive.google.com/file/d/1Hu-puF7YU50AH6Iq8IMJnC3Scag2Oov6/view).  

**Note:** the dataset can be accessed and downloaded only with a *unitn* account.  

## Prerequisites 

In order to run this project, the following tools have to be installed on your machine: 
- Python, preferably [3.8](https://www.python.org/downloads/release/python-380/) 

**Note:** Python version [3.9](https://www.python.org/downloads/release/python-390/) may not be supported by some modules (for further information and updates check [open3d](http://www.open3d.org/docs/latest/getting_started.html)).   

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

And then create and activate the virtual environment named *venv* typing in the command line (inside the project folder): 

- in Unix systems:
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

- in Windows systems:
    ```
    python -m venv venv
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

### Option 3: "images.pkl" and "labels.pkl"
Download the already-elaborated files [images.pkl](https://drive.google.com/file/d/1CdL_l_6IUfLe5UmuaQ9wPBmuMFW91Xo4/view?usp=sharing) and [labels.pkl](https://drive.google.com/file/d/1Hu-puF7YU50AH6Iq8IMJnC3Scag2Oov6/view?usp=sharing) and save them in the project directory. Do not uncomment any line.

### Run the script 
Having the virtual environment and the zip folder set up (either option 1 or 2), a last step must be performed. To start the entire architecture with the default hyperparameters, type in the command line: 

- in Unix systems:
    ```
    python3 main.py
    ```

- in Windows systems:
    ```
    python main.py
    ```
    
### Tuning the hyperparameters

The following hyperparameters can be customized by command-line arguments:

- **points**: Integer value representing the points sampled per Point Cloud. The default value is 1024. 
    ```
    python3 main.py -p 1024
    ```
- **db_sample**: Randomly sample the original dataset to obtain a subset of it. The default model consider all the cases and sets the parameter to 0.
    ```
    python3 main.py -s 0
    ```
- **train_test_split**: Floating number defining the train-test split size. The input value corresponds to the *test* size. The standard split 70/30 has been chosen as default. 
    ```
    python3 main.py -t 0.3
    ```
- **buffer**: Buffer size of training and testing loader. The default is set to 32, as reported in the reference paper. 
    ```
    python3 main.py -b 32
    ```
- **epochs**: Number of epochs to train the model. A relative low value, 100, is set as default and should be raised. 
    ```
    python3 main.py -e 100
    ```
- **learning_rate**: Starting learning rate for the model. It is worth notice that it will be halved every 20 epochs by the optimizer. The default value corresponds to 0.001
    ```
    python3 main.py -lr 0.001
    ```
- **feature_transform**: It activates the second tNet, namely the feature transformation network. It is set to **True** as default. 
    ```
    python3 main.py -ft True
    ```
- **balance**: It considers the class cardinalities and balance the dataset by introducing weights in the loss function. It is deactivated in the default model. 
    ```
    python3 main.py -bal False
    ```

An overview of all the hyperparameters and their description can be obtained by typing     
```diff
+ python3 main.py --help 
``` 

# 3D Object Classification Architecture with PointNet
![Architecture](https://github.com/LuciaHrovatin/AML-3D_object_classification/blob/dbd2a1ccf972f2cabdf8b28f9dda8cdc87e1ac29/docs/architecture.png)
The system transforms the initial scenes by splitting them into frames. Each Lego block is displayed in
identical frames: an RGB picture and a depth map. A Point Cloud is generated and initialises the PointNet classification network. It takes *n* points as input, applies input (and feature) transformations, aggregates point features by max-pooling and outputs the classification scores for the *k* classe

# Code structure

The backend code structure is composed by:
-   `data_ingestion.py`, containing the functions used to the cleansing procedure applied on the original dataset 
-   `data_splitter.py`, containing the functions for splitting the dataset into a training and a validation set 
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
├── docs
|   ├── architecture.png 
|   └── report.pdf 
└── main.py
```

