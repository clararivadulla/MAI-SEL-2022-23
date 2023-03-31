# PRISM
## Implementation of a rule based-algorithm

### Files & directories

The .zip file of this project contains the following subdirectories and files inside the main folder `PW1-PRISM`:

- `data`: contains the 4 selected datasets in .csv format. 
- `documentation`: contains the report of the project, with an explanation of the algorithm, its pseudo-code, how to execute the code, a discussion of the results and the conclusions. 
- `source`: with the `main.py` file, a `preprocessing.py` file (with the functions to read and preprocess the datasets), and the `prism.py` (where the algorithm is implemented). It also contains a `requirements.txt` file. 
- `source/results`: with the resulting rules and metrics (precision, coverage and recall) of every dataset in .csv format, as well as an `output.txt` file containing the output of executing the `main.py` file.
- A `README.md` file.

### Datasets

The datasets inside the `data` folder are:
- Congressional Voting Records
- Tic-Tac-Toe Endgame
- Car Evaluation
- Mushroom

### How to execute the code

Before running the code for the first time, you must change the path written in the 8th line
of the `main.py` file to the one where you’ve stored the folder containing the unzipped project.
Once the path is changed, you must follow the next steps:

1. Open the folder containing the code of the project (*source*) in the terminal `cd <root_folder_of_project>/source`
2. Create a virtual environment using Python `python3 -m venv venv/`
3. Open the virtual environment `source venv/bin/activate`
4. Install the required dependencies `pip install -r requirements.txt`
5. Run the main file of the project `python main.py`