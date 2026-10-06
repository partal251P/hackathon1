import csv
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("mortality_climate.csv")

df['season'] = df['season'].astype('category') # pour changer en categorical variable
df['CD_AGEGROUP'] = df['CD_AGEGROUP'].astype('category')

ordre = ["0-24", "25-44", "45-64", "65-74", "75-84", "85+"]

counts = df["CD_AGEGROUP"].value_counts().reindex(ordre)
df.info() 

plt.bar(counts, df["MS_NUM_DEATH"])
plt.show()

