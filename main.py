import csv
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("electricity_climate.csv")

# Datetime est de type object (texte) -> conversion en vraie date
df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
# On enlève les doublons et on met datetime comme index
df = df.sort_values("Datetime").drop_duplicates(subset="Datetime").set_index("Datetime")


lag = pd.Timedelta(days=7)

# Valeurs passées, poussées de 7 jours vers le futur pour s'aligner sur t
past = df[["Consumption", "temp_avg", "temp_avg_sq"]].copy()
past.index = past.index + lag # On change juste la date pour 7 après mais pas les données
past.columns = ["Cons_lag7", "T_lag7", "T_lag7_sq"] 