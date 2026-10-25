import math

import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t # For student t table


df = pd.read_csv("mortality_climate.csv")

df['season'] = df['season'].astype('category') # pour changer en categorical variable
df['CD_AGEGROUP'] = df['CD_AGEGROUP'].astype('category')

ordre = ["0-24", "25-44", "45-64", "65-74", "75-84", "85+"]

counts = df["CD_AGEGROUP"].value_counts().reindex(ordre)

    # 4.1) Create two subsets of the original dataset corresponding to the years 2022 and 2025, respectively.

sub22 = df[("2022-12-31" >= df["date"]) & (df["date"] >= "2022-01-01")]
sub25 = df[("2025-12-31" >= df["date"]) & (df["date"] >= "2025-01-01")]

    # 4.2) For the 85+ age group, perform a Student’s t-test to compare mortality between 2022 and 2025 **in summer**.
    # Check whether the assumptions required to apply this test are satisfied.

# Filter over age 85 and summer
sub22_summer_over_85 = sub22[(sub22["CD_AGEGROUP"] == "85+") & (sub22["season"] == "summer")]
sub25_summer_over_85 = sub25[(sub25["CD_AGEGROUP"] == "85+") & (sub25["season"] == "summer")]

# Find t-value = signal/noise = abs(mean(sub22...) - mean(sub25...))/sqrt(V[sub22...]/n1 + V[sub25...]/n2)
# (if unclear, go seek medical attention immediately)

std_sub22 = sub22_summer_over_85["MS_NUM_DEATH"].std()
std_sub25 = sub25_summer_over_85["MS_NUM_DEATH"].std()

mean_sub22 = sub22_summer_over_85["MS_NUM_DEATH"].mean()
mean_sub25 = sub25_summer_over_85["MS_NUM_DEATH"].mean() 

n1 = sub22_summer_over_85["MS_NUM_DEATH"].count()
n2 = sub25_summer_over_85["MS_NUM_DEATH"].count()

t_value = abs(mean_sub22 - mean_sub25) / math.sqrt((math.pow(std_sub22, 2)) / n1 + (math.pow(std_sub25, 2)) / n2)

alpha = 0.05 # Error margin
dof = n1 + n2 - 2 # Degree of liberty
t_compare = t.isf(alpha, dof)

print(f"t_value is {t_value}, student table is {t_compare}. t_value > t_compare, we REJECT the hypothesis H0")
# Differences between the data are NOT due to chance, there exists a factor influencing the mortality rate of the 85+ range of people
# between the summers of the years 2022 and 2025

# Assumptions for this test are : normal distribution, independence, similar variance, same number of data points and 20-30+ samples
# Independence... OK (someone cannot die in 2022 and 2025...)
# Normal distribution... OK
# Similar variance... OK (var_sub25 / var_sub22 ~= 1.68 <<< 4)
# Same number of data points... OK (94 >>> 20-30+)

# EVERY ASSUMPTION IS CORRECT, THE TEST IS ELIGIBLE !! #


    # 4.3) Explain what the Wilcoxon test is. Why could it be relevant in our setting?

    # 4.4) For the 85+ age group, perform a Wilcoxon test to compare mortality between 2022 and 2025 **in summer**.
    # What do you conclude?

    # 4.5) Perform the Wilcoxon test but now for the 25–44 age group and the **summer**.
    # What do you conclude in this case?