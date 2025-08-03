# ⚡ Electric Vehicle Recommender System

## 📁 Folder Structure
```
├── data
│ ├── 🥉 bronze # Contains raw, unprocessed data.
│ ├── 🥈 silver # Contains cleaned and transformed data.
│ └── 🥇 gold # Contains aggregated and feature-engineered data ready for modeling.
├── deploy
│ └── airflow
│ └── docker
│ └── Dockerfile
├── models
├── reports
├── src
│ ├── data_preprocessing.py
│ ├── evaluation.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ └── run_pipeline.py
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yaml
├── pyproject.toml
├── readme.md
---
```
## ⚙️ Setup Instructions

### 🐍 Setting Up the Python Environment using UV

```bash
# 1. Install uv
pip install uv

# 2. Create virtual environment using Python 3.12
uv venv .venv --python python3.12

# 3. Activate the virtual environment (Windows)
./venv/Scripts/activate

source .venv/bin/activate  # (Linux/Mac)

# 4. Initialize the environment
uv init

# 5. Add required packages
uv add scikit-learn
uv add pandas numpy
uv add ipykernel
```

## 💍 Pre-commit Hooks
To ensure code quality and consistency, we use pre-commit hooks. Follow these steps to set them up:
### 👰🏻‍♀️ Install pre-commit
```bash
uv add pre-commit
pre-commit install
pre-commit sample-config
replace .pre-commit-config.yaml hooks
  - id: check-yaml
    description: Check YAML files for syntax errors
  - id: end-of-file-fixer
    description: Ensure files end with a newline
  - id: trailing-whitespace
    description: Remove trailing whitespace
  - id: autopep8
    description: Format Python code using autopep8
pre-commit autoupdate
pre-commit run --all-files
```

## 🐳 DockerFile Initial Setup
1. Installed Docker to my machine
2. Pulled Airflow using `docker pull apache/airflow:2.9.3`
3. For a more organized Docker setup, create a folder named `deploy/docker` and place the `Dockerfile` inside it.
4. In the docker file, specify the python version, working directory, and any necessary environment variables/path.
5. Next, copy the necessary files e.g `pyproject.toml`, i found a command that translates pyproject.toml to requirements.txt, refer to the Dockerfile for more details.
6. Build the Docker image using the command:
```bash
docker build -f deploy/docker/Dockerfile -t 313ff744e86b92491bc16a84a7432558e43cbe39402a689339685c4fb16a0a02-ml-pipeline .
```
f flag specifies the path to the Dockerfile, and t flag specifies the name of the image.
7. Run the Docker container:
```bash
docker run --rm \
  313ff744e86b92491bc16a84a7432558e43cbe39402a689339685c4fb16a0a02-ml-pipeline
```
better to run the container in docker desktop since it shows the logs in real time and allows me to check the existing files in the container.


## 🐳 Docker Compose Setup
1. Create a `docker-compose.yaml` file in the root directory of your project.
  Pulled this from the Airflow documentation, it is a simple file that defines the services, networks, and volumes for your application.
2. Since I altered the location of airflow, i had to mount it properly in the docker-compose file under airflow-common>>volumes e.g `${AIRFLOW_PROJ_DIR:-.}/deploy/airflow/dags:/app/deploy/airflow/dags`
3. Another important configuration is the python path, to make the overall project structure work and simpler, i changed the PYTHONPATH to include the `/app` directory, which is where the main application code resides. This allows the Airflow services to access the Python modules in the `src` directory. /opt/airflow is also changed to /app/airflow
3. For the services, I exposed volume ./:/app so that the application code can be accessed by the Airflow services.
4. Docker File had to be updated and placed in the same directory as the docker-compose.yaml file.
5. I changed the base image to `apache/airflow:3.0.3` from `python:3.x.x` and had add user airflow. Because of this changes several dependencies had to be added to the `pyproject.toml` file, such as `apache-airflow`, `flask`, and `werkzeug`.




Chat GPT was consulted on how to setup Docker and Docker Compose for the project, including the necessary commands to build and run the Docker container. Used it to fast track the setup process while going through official documentation.
