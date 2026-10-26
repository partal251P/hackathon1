import csv
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("electricity_climate.csv")


# ------ Q5.1----------
# Datetime est de type object (texte) -> conversion en vraie date
df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
# On enlève les doublons et on met datetime comme index
df = df.sort_values("Datetime").drop_duplicates(subset="Datetime").set_index("Datetime")


lag = pd.Timedelta(days=7)

# Valeurs passées, poussées de 7 jours vers le futur pour s'aligner sur t
past = df[["Consumption", "temp_avg", "temp_avg_sq"]].copy()
past.index = past.index + lag # On change juste la date pour 7 après mais pas les données (c'est pour avoir les données avant et après les 7 jours)
past.columns = ["Cons_lag7", "T_lag7", "T_lag7_sq"] 


# On supprime la ligne d'après pour ne garder que la ligne qui nous intéresse
data = df[["Consumption"]].rename(columns={"Consumption": "Cons"}).join(past, how="inner")
# On supprime les lignes avec rien dedans  
data = data.dropna()