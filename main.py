import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from scipy.stats import poisson, nbinom 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 


# Load DataSet 
mortality_df = pd.read_csv("mortality_climate.csv")

# Convert into categorical variables 
mortality_df['season'] = mortality_df['season'].astype('category')
mortality_df['CD_AGEGROUP'] = mortality_df['CD_AGEGROUP'].astype('category')

# age order 
ordre_ages = ['0-24', '25-44', '45-64', '65-74', '75-84', '85+']
mortality_df['CD_AGEGROUP'] = pd.Categorical(mortality_df['CD_AGEGROUP'], categories=ordre_ages, ordered=True)

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












