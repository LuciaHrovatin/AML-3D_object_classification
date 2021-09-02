# Applied Machine Learning Project
## 3D Object Classification with PointNet

The **3D Object Classification with PointNet** project has been developed as final assignment of the Applied Machine Learning course, offered by the University of Trento.

## Project objective 
The project aims at deploying an architecture able to perform 3D Lego blocks extraction, transformation and classification. Specifically, the datasets employed have been cleaned and stored in a zip folder.   

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

### Upload the zip-folder 
This project employs only a dataset. 

### Run the script 
After the virtual environment and the zip folder set up, a last step must be manually performed. To start the entire architecture, type in the command line (with the activated virtual environment): 

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
-   `data_splitter.py`, containing the R scripts of the interface graphs and the Exploratory Data Analysis 
-   `src`, containing Python files with specific functions for data collection, data transformation and machine learning training/testing process 
-   `docker-compose.yml`, defining the Docker containers and their relationships  
-   `main.py`, triggers the entire project 

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
