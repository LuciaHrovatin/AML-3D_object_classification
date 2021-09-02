# Applied Machine Learning Project
## 3D Object Classification employing PointNet


## Project objective 

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
