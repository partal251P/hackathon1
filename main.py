import math
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from scipy.stats import poisson, nbinom, t, wilcoxon
import statsmodels.api as sm 
import statsmodels.formula.api as smf 


# Load DataSet 
mortality_df = pd.read_csv("mortality_climate.csv")

# Convert into categorical variables 
mortality_df['season'] = mortality_df['season'].astype('category')
mortality_df['CD_AGEGROUP'] = mortality_df['CD_AGEGROUP'].astype('category')

# age order 
age_order = ['0-24', '25-44', '45-64', '65-74', '75-84', '85+']
mortality_df['CD_AGEGROUP'] = pd.Categorical(mortality_df['CD_AGEGROUP'], categories=age_order, ordered=True)

# Summary 
mortality_df.info()

# plot number of death with respect to age_group 
mortality_df.groupby('CD_AGEGROUP')['MS_NUM_DEATH'].sum().plot(
    kind='bar',
    figsize=(10,6), 
    color='steelblue'
) # on aggrege MS_NUM_DEATH pour chaque groupe d'age 

plt.title("Nombre total de décès par tranche d'âge")
plt.xlabel("Tranche d'âge")
plt.ylabel("Nombre de décès")
plt.show()



plt.figure(figsize=(10, 6))

# A rajouter pour plus d'infos pas important (optionnel mais je trouve ca plus joli)
sns.boxplot(
    data=mortality_df, 
    x='CD_AGEGROUP', 
    y='MS_NUM_DEATH', 
    color='lightgray', 
)

# 2. Superposer tous les points par-dessus
sns.stripplot(
    data=mortality_df, 
    x='CD_AGEGROUP', 
    y='MS_NUM_DEATH', 
    jitter=0.2, 
    alpha=0.4, 
    color='crimson', 
    size=2
)

plt.title("Distribution et statistiques des décès par tranche d'âge")
plt.xlabel("Tranche d'âge")
plt.ylabel("Nombre de décès")

plt.show()


#### Q2 #####
# on adapte les données 
sub_data_set = mortality_df[(mortality_df['CD_AGEGROUP']== '45-64') & (mortality_df['season'] == 'winter')].copy()
data = sub_data_set['MS_NUM_DEATH'].dropna() # rejette les valeurs manquantes 



# -------------------------------------------------------
# ajustement par maximum de vraisemblance (MLE)
# -------------------------------------------------------

# A) distribution de poisson 
# le parametre lambda pour une MLE est la moyenne empirique 
lambda_mle = data.mean()

# log-likekyhood poisson 
log_likelyhood_poisson = np.sum(poisson.logpmf(data, mu=lambda_mle))

# B) Distribution binomiale négative (version corrigée)
model_nb = smf.negativebinomial("MS_NUM_DEATH ~ 1", data=sub_data_set).fit(disp=0)

# on extrait les données qu'on vient de calculer (logique j'ai envie de dire)
mu_nb = model_nb.predict()[0]
alpha_nb = model_nb.params['alpha']  # faut lire la doc c'était long

# Adaptation des paramètres pour scipy.stats.nbinom (ia utilisée pour comprendre d'ou ca vient)
n_param = 1 / alpha_nb
p_param = n_param / (n_param + mu_nb)

# Log-likelihood pour binomiale négative
log_likelyhood_neg_binomial = np.sum(nbinom.logpmf(data, n=n_param, p=p_param))

print(f"--- RÉSULTATS DE L'AJUSTEMENT (MLE) ---")
print(f"Poisson λ_hat = {lambda_mle:.4f}")
print(f"Poisson Log-Likelihood: {log_likelyhood_poisson:.2f}\n")

print(f"Negative Binomial μ = {mu_nb:.4f}, α = {alpha_nb:.4f}")
print(f"Negative Binomial (n={n_param:.4f}, p={p_param:.4f})")
print(f"Negative Binomial Log-Likelihood: {log_likelyhood_neg_binomial:.2f}\n")

if log_likelyhood_neg_binomial > log_likelyhood_poisson:
    print("Verdict : La distribution Binomiale Négative offre une meilleure adéquation (Log-Likelihood plus élevée).")
else:
    print("Verdict : La distribution de Poisson offre une meilleure adéquation.")


# tous les prints du dessus proviennent de l'ia, permet de debuguer et visualiser les résultats proprement 




# ---------------------------------------------------------
# 4. Graphique : Histogramme et PMF superposées
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Histogramme empirique (densité / fréquence)
k_values = np.arange(data.min(), data.max() + 1)
counts, bins = np.histogram(data, bins=np.append(k_values, k_values[-1] + 1) - 0.5, density=True)
ax.bar(k_values, counts, width=0.8, alpha=0.4, color='gray', label='Données empiriques (Winter, 45-64)')

# PMF Poisson
pmf_poisson = poisson.pmf(k_values, mu=lambda_mle)
ax.plot(k_values, pmf_poisson, 'o--', color='dodgerblue', linewidth=2, label=f'Poisson (λ={lambda_mle:.2f})')

# PMF Binomiale Négative
pmf_nb = nbinom.pmf(k_values, n=n_param, p=p_param)
ax.plot(k_values, pmf_nb, 's-', color='crimson', linewidth=2, label=f'Binomiale Négative (n={n_param:.2f}, p={p_param:.2f})')

# Habillage du graphique
ax.set_title("Ajustement des distributions Poisson et Binomiale Négative (MLE)\nTranche 45–64 ans - Hiver", fontsize=13, fontweight='bold')
ax.set_xlabel("Nombre de décès (MS_NUM_DEATH)", fontsize=11)
ax.set_ylabel("Probabilité / Densité", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Le dernier graphe a été rendu plus joli par ia aussi, fonctionnait avant mais moins lisible que mtn 


df = pd.read_csv("mortality_climate.csv")

df['season'] = df['season'].astype('category') # pour changer en categorical variable
df['CD_AGEGROUP'] = df['CD_AGEGROUP'].astype('category')

ordre = ["0-24", "25-44", "45-64", "65-74", "75-84", "85+"]

counts = df["CD_AGEGROUP"].value_counts().reindex(ordre)

    # 4.1) Create two subsets of the original dataset corresponding to the years 2022 and 2025, respectively.

sub22 = df[("2022-12-31" >= df["date"]) & (df["date"] >= "2022-01-01")]
sub25 = df[("2025-12-31" >= df["date"]) & (df["date"] >= "2025-01-01")]

print(f"\033[1;34m 2022 subset is \033[0;0m \n{sub22},\n\n \033[1;34m2025 subset is\033[0;0m \n{sub25}")

    # 4.2) For the 85+ age group, perform a Student’s t-test to compare mortality between 2022 and 2025 **in summer**.
    # Check whether the assumptions required to apply this test are satisfied.

# Filter over age 85 and summer
sub22_summer_over_85 = sub22[(sub22["CD_AGEGROUP"] == "85+") & (sub22["season"] == "summer")]
sub25_summer_over_85 = sub25[(sub25["CD_AGEGROUP"] == "85+") & (sub25["season"] == "summer")]

print(f"\033[1;34m2022 subset filtered by age > 85+ during summer is\033[0;0m \n{sub22_summer_over_85},\n\n \033[1;34m2025 subset filtered by age > 85+ during summer is\033[0;0m \n{sub25_summer_over_85}")

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

print("""The t_value for the two subsets above is {0:0.5f}.
The value of the student table with alpha = 0.05 and degree of liberty 94 + 94 - 2 = 186 is {1:0.5f}.
We can see that t_value > t_compare, we REJECT the hypothesis H0""".format(t_value, t_compare))
# Differences between the data are NOT due to chance, there exists a factor influencing the mortality rate of the 85+ range of people
# between the summers of the years 2022 and 2025

# Assumptions for this test are : normal distribution, independence, similar variance, same number of data points and 20-30+ samples
# Independence... OK (someone cannot die in 2022 and 2025...)
# Normal distribution... OK
# Similar variance... OK (var_sub25 / var_sub22 ~= 1.68 <<< 4)
# Same number of data points... OK (94 >>> 20-30+)

# EVERY ASSUMPTION IS CORRECT, THE TEST IS ELIGIBLE !! #


    # 4.3) Explain what the Wilcoxon test is. Why could it be relevant in our setting?

# Similar to t_test BUT data does not have to be normally distributed (non parametric). It assumes a certain symmetry of values
# around the median. It's useful especially when data are not necessarily normally distributed.
# In our context, a Student t_test would be more appropriate because every assumption is fulfilled. If it wasn't, then a non parametric
# test such as this one would be more appropriate.

    # 4.4) For the 85+ age group, perform a Wilcoxon test to compare mortality between 2022 and 2025 **in summer**.
    # What do you conclude?

wilcoxon_85 = wilcoxon(sub22_summer_over_85["MS_NUM_DEATH"], sub25_summer_over_85["MS_NUM_DEATH"])
print(f"The p_value of the Wilcoxon test for the two subsets of question 4.2 is {wilcoxon_85.pvalue}")
# The p_value (0.0017383269781142687) is lower than the chosen alpha (0.05). Thus, we reject H0 because there is a significant difference
# between the two medians of both the samples.

    # 4.5) Perform the Wilcoxon test but now for the 25–44 age group and the **summer**.
    # What do you conclude in this case?

sub22_summer_25_44 = sub22[(sub22["CD_AGEGROUP"] == "25-44") & (sub22["season"] == "summer")]
sub25_summer_25_44 = sub25[(sub25["CD_AGEGROUP"] == "25-44") & (sub25["season"] == "summer")]

wilcoxon_25_44 = wilcoxon(sub22_summer_25_44["MS_NUM_DEATH"], sub25_summer_25_44["MS_NUM_DEATH"])
print(f"""wilcoxon_25_44.pvalue""")
print(f"""2022 subset filtered by age between 25 and 44 during summer is \n{sub22_summer_25_44},
        \n 2025 subset filtered by age between 25 and 44 during summer is \n{sub25_summer_25_44}.
        
        The p_value of the Wilcoxon test for the two subsets is {wilcoxon_25_44.pvalue}""")
# The p_value (0.6369234788432132) is *higher* than the chosen alpha (0.05). Thus, we accept H0 because there is no significant difference
# between the two medians of both the samples.
