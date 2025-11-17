# source .venv/bin/activate

import pandas as pd
import matplotlib as plt

df = pd.read_csv("./nba.csv")

df[df["Age"]>30] # Jugadores con edad mayor que 30

df["FromKentucky"] = df['College']=="Kentucky" # Añadir nueva columna al dataframe

res = df[["Age", "Salary"]].mean() 

print(res)