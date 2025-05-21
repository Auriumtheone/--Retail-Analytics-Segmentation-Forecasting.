import os

# Projekto aplankai
folders = [
    "data/raw", "data/processed",
    "models", "api", "dashboard",
    "notebooks", "tests", "docs", "utils"
]

# Sukurti aplankus, jei jų dar nėra
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Sukurti tuščius failus ateičiai
files = {
    "models/kmeans_segment.py": "",
    "models/lstm_forecasting.py": "",
    "models/arima_model.py": "",
    "models/model_comparison.py": "",
    "api/app.py": "",
    "api/routes.py": "",
    "api/database.py": "",
    "dashboard/dash_layout.py": "",
    "dashboard/callbacks.py": "",
    "dashboard/visuals.py": "",
    "notebooks/project_overview.ipynb": "",
    "notebooks/data_exploration.ipynb": "",
    "notebooks/model_training.ipynb": "",
    "notebooks/evaluation.ipynb": "",
    "tests/test_models.py": "",
    "tests/test_api.py": "",
    "docs/README.md": "",
    ".gitignore": "",
    "requirements.txt": "",
    "run.py": ""
}

# Sukurti tuščius failus
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print(" Projektinė struktūra sukurta!")