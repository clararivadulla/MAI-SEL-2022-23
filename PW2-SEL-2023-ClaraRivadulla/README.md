# Decision forest & Random forest
## Combining multiple classifiers

### Files & directories

The .zip file of this project contains the following subdirectories and files inside the main folder:

- `data`: contains the selected datasets in .csv format (_Iris_, _Tic-Tac-Toe Endgame_, _Abalone_ and _Mushroom_). 
- `documentation`: contains the report of the project, with an explanation of the algorithms, its pseudo-codes, how to execute the code, a discussion of the results and the conclusions. 
- `source`: contains the `main.py` file, a `preprocessing.py` file (with the functions to read and preprocess the datasets), `decision_forest.py` and `random_forest.py` files (where the Decision Forest and Random Forest algorithms are implemented along with the class `DecisionTree`), a `node.py` file where the node of the decision trees is defined, and a `plots.py` file that's used to save the plots with the resulting accuracies for each method and dataset. Inside `source` there are also a `results` folder with the results saved in `.csv` format and a `figures` folder with the resulting plots from executing `plots.py`.  
- A `README.md` file.

### Datasets

The datasets inside the `data` folder are:
- Iris
- Tic-Tac-Toe Endgame
- Abalone
- Mushroom

### How to execute the code

Before running the code for the first time, you must change the path written in the **12th line**
of the `main.py` file to the one where you’ve stored the folder containing the unzipped project.
Once the path is changed, you must follow the next steps:

1. Open the folder containing the code of the project (*source*) in the terminal `cd <root_folder_of_project>/source`
2. Create a virtual environment using Python `python3 -m venv venv/`
3. Open the virtual environment `source venv/bin/activate`
4. Install the required dependencies `pip install -r requirements.txt`
5. Run the main file of the project `python main.py`