Setting up script for Data Analysis

# Prerequisites

## Make sure you have the following installed:

* Python (3.8 or newer)
* Visual Studio Code
* pip (comes with Python)

## To check Python installation:

* python --version
or
* python3 --version

# Follow the step to run Jupyter notebook in VSC
## 🛠 Step 1: Install Required VS Code Extensions

* Open VS Code → Extensions (Ctrl+Shift+X) and install:
	* Python (by Microsoft)
	* Jupyter (by Microsoft)

Restart VS Code after installation.

## 🐍 Step 2: Select Your Installed Python Interpreter

* Open VS Code
* Press Ctrl+Shift+P (macOS: Cmd+Shift+P)
* Search and select:
	* Python: Select Interpreter
* Choose your installed Python (example):
	* /usr/bin/python3
	* ~/miniforge3/bin/python
	* C:\Python311\python.exe

✔ This Python will be used by Jupyter.

## 📦 Step 3: Install Jupyter in That Python Environment

* Open VS Code Terminal:
* Click on Terminal → New Terminal
* Run:
	* pip install jupyter notebook ipykernel
* In terminal verify:
	* jupyter --version

## 📓 Step 4: Open or Create a Jupyter Notebook
* Option A: Open Existing Notebook
	* code example.ipynb
* Option B: Create New Notebook
	* In VS Code → File → New File
	* Save as: my_notebook.ipynb

VS Code will automatically open it in Jupyter mode.

## ⚙️ Step 5: Select Python Kernel for Notebook

* Open the .ipynb file
* Click Kernel Selector (top-right)
* Choose:
	* Python (your-selected-interpreter)
	- for example: Python 3.11 (miniforge)

✔ Your notebook is now linked to your installed Python.

## ▶️ Step 6: Run Notebook Cells

* Run single cell: Shift + Enter
* Run all cells: Run → Run All
* Restart kernel: Kernel → Restart Kernel

## 🧪 Step 7: Test Installation

* Run this in a notebook cell:
	- import sys
	- print(sys.executable)

Output should match your selected Python path.


# Installing python dependencies to run script

All the that are needed to run the data analysis script have been listed out in the 
requirements file. TO install these dependencies, run the following command: 

 - pip3 install -r ~/Desktop/Github/molecular_fmri_data_analysis/requirements.txt