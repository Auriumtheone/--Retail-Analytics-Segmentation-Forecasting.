import os

# Aplankai, kurių **visai** nerodysime
EXCLUDE_DIRS = {"__pycache__", "venv", "new_env", ".git", ".idea", "dist", "build", ".venv", "site-packages"}

# Failų tipai, kurių **visai** nerodysime
EXCLUDE_FILES_EXTENSIONS = {".pyc", ".log", ".tmp", ".cache", ".txt", ".json"}

def print_directory_structure(start_path, indent=0):
    for item in sorted(os.listdir(start_path)):
        item_path = os.path.join(start_path, item)

        # Visiškai praleidžiame nurodytus aplankus
        if os.path.isdir(item_path) and item in EXCLUDE_DIRS:
            continue

        if os.path.isdir(item_path):
            print("│   " * indent + "├── " + item + "/")
            print_directory_structure(item_path, indent + 1)
        else:
            # Visiškai praleidžiame failus su nurodytomis galūnėmis
            if any(item.endswith(ext) for ext in EXCLUDE_FILES_EXTENSIONS):
                continue
            print("│   " * indent + "├── " + item)

if __name__ == "__main__":
    start_directory = "."  # Čia gali nurodyti konkretų aplanką
    print("Retail Analytics Folder Structure:")
    print_directory_structure(start_directory)