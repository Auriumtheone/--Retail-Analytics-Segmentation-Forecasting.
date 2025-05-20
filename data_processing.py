import pandas as pd
import numpy as np

# Funkcija duomenų įkėlimui
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Funkcija duomenų struktūros peržiūrai
def explore_data(df):
    print("\n Duomenų struktūra:")
    print(df.info())
    print("\n Statistinė santrauka:")
    print(df.describe())

# Funkcija trūkstamų reikšmių tvarkymui
def clean_data(df):
    # Užpildyti trūkstamas reikšmes su medianomis
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

# Paleidimo blokas (jei skriptas vykdomas tiesiogiai)
if __name__ == "__main__":
    file_path = "data/raw/online_retail_II.xlsx"
    
    # Įkelti duomenis
    df = load_data(file_path)

    # Atlikti duomenų analizę
    explore_data(df)

    # Apdoroti duomenis
    df = clean_data(df)
    print("\n Duomenys išvalyti ir paruošti!")