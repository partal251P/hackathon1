import csv
import pandas as pd
import matplotlib.pyplot as plt


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

# 4.3) Explain what the Wilcoxon test is. Why could it be relevant in our setting?

# 4.4) For the 85+ age group, perform a Wilcoxon test to compare mortality between 2022 and 2025 **in summer**.
# What do you conclude?

# 4.5) Perform the Wilcoxon test but now for the 25–44 age group and the **summer**.
# What do you conclude in this case?