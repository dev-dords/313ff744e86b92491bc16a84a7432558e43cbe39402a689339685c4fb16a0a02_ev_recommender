## Data Structure
#### Bronze
Description: Contains raw, unprocessed data.

#### Silver
Description: Contains cleaned and transformed data.

## Setup Instructions

### Installing UV
```bash
1. pip install uv
2. uv venv .venv --python python3.12
3. ./venv/Scripts/activate since i am on windows. Run source ./venv/Scripts/activate
4. uv init
5. uv add scikit-learn
6. uv add pandas, numpy
7. uv add ipykernel
```

### Some notes
1. Initially not clearing notebook upon commit but remembered there is a command run during the lectures to clear the outputs for the notebook.
2. Since I implemented a recommender system, No metrics are used to evaluate the model. The recommender system is evaluated based on user feedback and satisfaction.
