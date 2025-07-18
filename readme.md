# ⚡ Electric Vehicle Recommender System

## 📁 Data Structure

### 🥉 Bronze
- **Description:** Contains raw, unprocessed data.

### 🥈 Silver
- **Description:** Contains cleaned and transformed data.

### 🥇 Gold
- **Description:** Contains aggregated and feature-engineered data ready for modeling.
---

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

## Pre-commit Hooks
To ensure code quality and consistency, we use pre-commit hooks. Follow these steps to set them up:
### 1. Install pre-commit
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
