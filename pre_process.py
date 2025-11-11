# %% [markdown]
# <h1>Préprocessing</h1>
# 
# <h3>Lecture du dataset</h3>
# 
# Dans un premier temps, nous ouvrons simplement le fichier 'welddb.data', et nous nommons les colonnes selon le fichier .info

# %%
import pandas as pd

df = pd.read_csv("./welddb/welddb.data", 
                 sep=r"\s+",      
                 header=None,    
                 na_values="N")  

columns = [
    "C", "Si", "Mn", "S", "P", "Ni", "Cr", "Mo", "V", "Cu", "Co", "W",
    "O", "Ti", "N", "Al", "B", "Nb", "Sn", "As", "Sb",
    "Current", "Voltage", "AC_DC", "Polarity", "Heat_input", "Interpass_temp",
    "Weld_type", "PWHT_temp", "PWHT_time",
    "Yield_strength", "UTS", "Elongation", "RA",
    "Charpy_temp", "Charpy_toughness", "Hardness",
    "FATT50", "Primary_ferrite", "Ferrite_2nd_phase",
    "Acicular_ferrite", "Martensite", "Ferrite_carbide",
    "Weld_ID"
]

df.columns = columns

# %% [markdown]
# Nous nous intéressons ensuite à la structure de la base de données, nous utlisons donc les méthodes fournies par pandas pour analyser les différentes colonnes, ainsi que déterminer lesquelles ont le plus de valeurs non renseignées.

# %%
df.describe()

# %%
df.info()

# %%
(df.isnull().sum().sort_values(ascending=False)/df.shape[0])*100

# %% [markdown]
# <h3>Nettoyage des données</h3>

# %% [markdown]
# Dans un second temps, nous nettoyons les données pour supprimer les valeurs non numériques dans les colonnes censées l'être.

# %%
import re
import numpy as np

def nettoyer_valeur(val):
    val = str(val).strip()

    # Cas 'N' ou vide
    if val.upper() == "N" or val == "":
        return np.nan

    # Cas '<5' -> remplacer par 5 (ou NaN si vous préférez)
    if val.startswith("<"):
        try:
            return float(val[1:])
        except:
            return np.nan

    # Cas '67tot33res' -> extraire ce qu'il y a avant 'tot'
    match = re.match(r"(\d+)", val)
    if match:
        return float(match.group(1))

    return np.nan

colonnes_a_nettoyer = [
    'C', 'Si', 'Mn', 'S', 'P', 'Ni', 'Cr', 'Mo', 'V', 'Cu', 
    'Co', 'W', 'O', 'Ti', 'N', 'Al', 'B', 'Nb', 'Sn', 'As', 'Sb', 'Primary_ferrite'
]

# On applique la fonction seulement sur les colonnes ciblées
df[colonnes_a_nettoyer] = df[colonnes_a_nettoyer].applymap(nettoyer_valeur)

# %%
df['Interpass_temp'] = df['Interpass_temp'].replace("150-200", 175)
df['Interpass_temp'] = pd.to_numeric(df['Interpass_temp'])
print(df.info())

# %%
df["Hardness"] = df["Hardness"].astype(str).str.extract(r"^(\d+\.?\d*)").astype(float)
df['Hardness'] = pd.to_numeric(df['Hardness'])

# %%
df.info()

# %% [markdown]
# On observe que toutes les colonnes devant contenir des valeurs numériques sont bien au format numérique. Nous allons maintenant nous ocuper des variables catégorielles.

# %%
df.info()

# %%
df = df.drop('Weld_ID', axis=1)

# %%
df.info()

# %% [markdown]
# <h3>Colonnes à supprimer</h3>
# 
# Nous allons maintenant nou intéresser aux features que nous pouvons supprimer de notre dataset sans perdre trop d'information. Pour ce faire, nous allons étudier la corrélation entre les variables, et supprimer celles pour lesquelles les données sont largement absentes, tout en ayant très peu de corrélation avec les variables d'intérêt pour qualifier la qualité d'une soudure. D'après le papier scientifique fourni avec le dataset, les trois features les plus intéressantes pour qualifier la qualité 'une soudure sont les suivantes : 
#     <li>Charpy Toughness</li>
#     <li>Elongation</li>
#     <li>Ultimate tensile strength</li>

# %%
miss = df.isna().mean().sort_values(ascending=False)
miss_df = miss.to_frame(name="missing_ratio").reset_index().rename(columns={"index": "column"})

# %%
import plotly.express as px

target_vars = ["Charpy_toughness", "Elongation", "UTS", "Yield_strength"]

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

corrs = df[num_cols].corr(method='pearson')[target_vars]

# Tri des corrélations en fonction de la quantité de missing values
corrs_sorted = corrs.reindex(miss_df['column'])


print("=== Corrélation absolue avec les variables de qualité ===")
print(corrs_sorted.round(3))
fig = px.imshow(
    corrs_sorted,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Correlation absolue entre les variables numériques et les variables de qualité"
)
fig.show()

# %% [markdown]
# Suite à cette étude, nous élimonons les colonnes suivantes :

# %%
COLS_TO_DROP = ["FATT50", "W", "Ferrite_carbide", "Martensite", "Ferrite_2nd_phase", "Acicular_ferrite", 
                "Primary_ferrite", "Co", "Hardness", "As", "Sb", "Sn", "B", "Cu"]
df = df.drop(columns=COLS_TO_DROP)

# %%
df.info()

# %% [markdown]
# D'après le même papier scientifique, nous pouvons aussi remplacer les valeurs vides de certaines colonnes par 0.

# %% [markdown]
# Ensuite, pour remplir les autres colonnes, nous utilisons un KNNImputer de la bibliothèque sklearn, de sorte à peupler notre dataset avec des valeurs potentiellement plus proches de la réalité que par des transformation utilisant la médiane ou la moyenne de l'ensemble du dataset.

# %% [markdown]
# Pour être en mesure d'évaluer la qualité de notre modèle, nous séparons tout de suite le dataset en 2 (un set pour le train et un set pour le test). Ainsi, nous remplirons uniquement les valeurs manquantes pour le dataset de train à l'aide d'un KNN.

# %%
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df,random_state=42,train_size=0.8)

print(df_test.shape)
print(df_train.shape)

df_train.info()

# %%
impurity_elements = ["P", "S"]
deliberate_elements = ["Mn", "Ni", "Cr", "Mo"]

for col in impurity_elements + deliberate_elements:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

# Add indicators
for col in impurity_elements + deliberate_elements:
    df_train[f"{col}_was_na"] = df_train[col].isna().astype(int)

# Apply domain-specific fills
for col in impurity_elements:
    df_train[col] = df_train[col].fillna(df_train[col].mean())
for col in deliberate_elements:
    df_train[col] = df_train[col].fillna(0)

# %%
#uts before imputation :
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("Nombre total de valeurs manquantes avant traitement :", df_train.isna().sum().sum())

px.histogram(df_train, x="Ti", nbins=30, title="Ti before imputation").show()

# Colonnes à exclure du traitement
excluded_cols = ["Charpy_toughness", "Elongation", "UTS"]

# Séparation des colonnes numériques et catégorielles
num_cols = df_train.select_dtypes(include=[np.number]).columns
cat_cols = df_train.select_dtypes(exclude=[np.number]).columns

# Colonnes numériques sur lesquelles appliquer le KNNImputer
num_cols_to_impute = [col for col in num_cols if col not in excluded_cols]

# Création du pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=5))
])

# Application du pipeline uniquement sur les colonnes sélectionnées
df_train[num_cols_to_impute] = pipeline.fit_transform(df_train[num_cols_to_impute])

# Remplissage des valeurs manquantes pour les variables catégorielles
for col in cat_cols:
    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

print(cat_cols)
print(df_train.info())
# Vérification finale du nombre de valeurs manquantes restantes
print("Nombre total de valeurs manquantes après traitement :", df_train.isna().sum().sum())
px.histogram(df_train, x="Ti", nbins=30, title="Ti after imputation").show()

# %%
df_train.info()

# %%
#convert AC/DC to 1/0
df_train['AC_DC'] = df_train['AC_DC'].map({'AC': 1, 'DC': 0})

dummy_cols = [c for c in df_train.columns if c.startswith('Polarity_') or c.startswith('Weld_type_')]
df_train[dummy_cols] = df_train[dummy_cols].astype(int)

# %% [markdown]
# <h3>PCA</h3>
# 
# Nous réalisons ensuite une PCA pour faire apparaitre quelles sont les features les plus intéressantes pour qualifier la qualité d'une soudure, en excluant les 3 colonnes citées précédemment.

# %%
df_train.info()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------
# 0) Paramètres utilisateur
# -------------------------
excluded_cols = ["Charpy_toughness", "Elongation", "UTS", "P_was_na", "S_was_na", "Mn_was_na", "Ni_was_na", "Cr_was_na", "Mo_was_na", "Weld_type_FCA",
                 "Weld_type_GMAA", "Weld_type_GTAA", "Weld_type_MMA", "Weld_type_NGGMA", "Weld_type_NGSAW", "Weld_type_SA", "Weld_type_SAA", 
                 "Weld_type_ShMA", "Weld_type_TSA", "Polarity_0", "Polarity_+", "Polarity_-"]
label_col = None  # (optionnel) nom d'une colonne à afficher comme étiquette sur le scatter PC1–PC2 (ex: "AlloyID")
n_top_loadings = 15  # nombre de variables les plus contributrices à afficher

# -------------------------
# 1) Préparation des données
# -------------------------
num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
pca_features = [c for c in num_cols if c not in excluded_cols]


# Standardisation
scaler = StandardScaler()  ## attention on a deja scale
X_scaled = scaler.fit_transform(df_train[pca_features])

# -------------------------
# 2) PCA
# -------------------------
pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

# Noms des composantes et DataFrames utiles
pc_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df_scores = pd.DataFrame(X_pca, columns=pc_names, index=df_train.index)
loadings = pd.DataFrame(pca.components_.T, index=pca_features, columns=pc_names)

# -------------------------
# 3) Graphiques
# -------------------------

# (A) Scree plot — variance expliquée par composante
plt.figure()
plt.plot(range(1, len(explained) + 1), explained, marker='o')
plt.title("Scree plot — Variance expliquée par composante")
plt.xlabel("Composante principale")
plt.ylabel("Part de variance expliquée")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# (B) Variance expliquée cumulée
plt.figure()
plt.plot(range(1, len(cum_explained) + 1), cum_explained, marker='o')
plt.title("Variance expliquée cumulée")
plt.xlabel("Composante principale")
plt.ylabel("Variance expliquée cumulée")
plt.axhline(0.80, linestyle='--')  # repère 80%
plt.axhline(0.90, linestyle='--')  # repère 90%
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# (C) Nuage de points PC1 vs PC2
plt.figure()
plt.scatter(df_scores["PC1"], df_scores["PC2"])
plt.title("Projection des observations — PC1 vs PC2")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% de variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% de variance)")
plt.grid(True, linestyle='--', alpha=0.5)



plt.tight_layout()
plt.show()

# (D) Contributions (loadings) — variables les plus importantes sur PC1
# On affiche les |coefficients| les plus élevés de PC1
pc1_load = loadings["PC1"].abs().sort_values(ascending=False).head(n_top_loadings)
vars_to_plot = pc1_load.index.tolist()

plt.figure(figsize=(8, max(4, 0.3*len(vars_to_plot))))
plt.barh(vars_to_plot[::-1], loadings.loc[vars_to_plot, "PC1"][::-1])
plt.title(f"Contributions des variables à PC1 (top {len(vars_to_plot)})")
plt.xlabel("Coefficient de chargement (loading)")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()

# -------------------------
# 4) (Option) Export des résultats
# -------------------------
# df_scores.to_csv("pca_scores.csv", index=False)
# loadings.to_csv("pca_loadings.csv")






# %% [markdown]
# <h2>Modélisation : régressions multi‑cibles (UTS, Allongement, Charpy, Re)</h2>
# 
# Objectif : prédire la qualité d'une soudure via 4 variables continues :
# - UTS (Ultimate Tensile Strength)
# - Elongation
# - Charpy_toughness
# - Yield_strength
# 
# Approche : on construit un pipeline sklearn complet (pré‑traitement + modèle) pour éviter toute fuite de données.
# On évalue par validation croisée (KFold) puis on fait un hold‑out sur un jeu de test.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


TARGETS = ["UTS", "Elongation", "Charpy_toughness", "Yield_strength"]
TARGETS = [t for t in TARGETS if t in df.columns]

# On recommence une découpe propre train/test depuis df (après nettoyage et drops précédents)
# IMPORTANT : les modèles ne gèrent pas les NaN dans y -> on supprime les lignes sans cibles complètes
if len(TARGETS) == 0:
    raise ValueError("Aucune cible disponible dans le DataFrame 'df'. Vérifiez les noms de colonnes.")

missing_y_mask = df[TARGETS].isna().any(axis=1)
print(f"Lignes avec cibles manquantes : {missing_y_mask.sum()} (elles seront supprimées pour l'entraînement)")

df_model = df.loc[~missing_y_mask].copy()
X = df_model.drop(columns=TARGETS)
y = df_model[TARGETS].copy()

# Split reproductible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Détection des colonnes numériques et catégorielles
num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = [c for c in X_train.columns if c not in num_features]

#Préprocesseurs
num_transformer = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    # StandardScaler sur matrices creuses -> with_mean=False
    ("scaler", StandardScaler(with_mean=False))
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features),
])


# RandomForest et ExtraTrees sont deux forêts d'arbres de décision,
# mais ils diffèrent sur la manière de choisir les coupures (splits) :
#  - RandomForest : pour chaque nœud, on sélectionne un sous‑ensemble aléatoire
#    de features (max_features) puis on cherche LA meilleure coupure (gain impurité max)
#    sur ces features. Échantillonnage par bootstrap en général.
#  - ExtraTrees (Extremely Randomized Trees) : on sélectionne un sous‑ensemble aléatoire
#    de features ET on tire aléatoirement plusieurs seuils de coupe par feature, puis on choisit
#    parmi ces seuils aléatoires celui qui donne le meilleur gain. Moins de variance, plus de
#    biais, souvent plus robuste dans les petits jeux tabulaires et plus rapide.
# Conséquence pratique : ExtraTrees lisse mieux le bruit en général, à overfitter moins souvent
# que RandomForest, et donne fréquemment de meilleurs scores de généralisation sur données tabulaires
# bruyantes/peu nombreuses — ce que l’on va observer ici plus tard.
rf = Pipeline([
    ("prep", preprocessor),
    ("model", MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )
    ))
])

et = Pipeline([
    ("prep", preprocessor),
    ("model", MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=600,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )
    ))
])


cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

results = {}
for name, pipe in {"RandomForest": rf, "ExtraTrees": et}.items():
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    results[name] = {
        "MAE_cv_mean": -np.mean(cv_res["test_MAE"]),
        "RMSE_cv_mean": -np.mean(cv_res["test_RMSE"]),
        "R2_cv_mean": np.mean(cv_res["test_R2"]),
    }

print("=== Validation croisée (moyenne 5‑fold) ===")
for k, v in results.items():
    print(f"{k:>12} | MAE={v['MAE_cv_mean']:.3f} | RMSE={v['RMSE_cv_mean']:.3f} | R2={v['R2_cv_mean']:.3f}")

# Sélection du meilleur modèle simple (au R2 moyen)
best_name = max(results.keys(), key=lambda n: results[n]["R2_cv_mean"])
best_pipe = rf if best_name == "RandomForest" else et
print(f"\nMeilleur modèle d'après la CV : {best_name}")

# Entraînement final sur train et évaluation hold‑out
best_pipe.fit(X_train, y_train)

# Prédictions et métriques globales multi‑sorties
pred = best_pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred, multioutput="uniform_average")
mse = mean_squared_error(y_test, pred, multioutput="uniform_average")
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred, multioutput="uniform_average")
print("\n=== Hold‑out sur jeu de test ===")
print(f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")

# On affiche les détails par cible (utile pour l'analyse comparative)
print("\nDétail par variable cible :")
for i, col in enumerate(TARGETS):
    mae_c = mean_absolute_error(y_test.iloc[:, i], pred[:, i])
    mse_c = mean_squared_error(y_test.iloc[:, i], pred[:, i])
    rmse_c = np.sqrt(mse_c)
    r2_c = r2_score(y_test.iloc[:, i], pred[:, i])
    print(f" - {col:>16} | MAE={mae_c:.3f} | RMSE={rmse_c:.3f} | R2={r2_c:.3f}")

# Export des prédictions de test (pour figures/rapport)
proba_df = pd.DataFrame(pred, columns=[f"pred_{c}" for c in TARGETS], index=y_test.index)
report_df = pd.concat([y_test.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
report_df.to_csv("./welddb_predictions_holdout.csv", index=False)

print("\nPrédictions sauvegardées dans welddb_predictions_holdout.csv")

# %% [markdown]
# <h2>Modélisation par cible (exploit all available labels per target)</h2>
# Même prétraitement et mêmes modèles, mais on entraîne une pipeline par variable cible
# en utilisant toutes les lignes où cette cible est renseignée.

from sklearn.model_selection import RepeatedKFold

SINGLE_TARGETS = [t for t in ["UTS", "Elongation", "Charpy_toughness", "Yield_strength"] if t in df.columns]

print("\n=== Entraînement par cible (jeu propre à chaque target) ===")

per_target_reports = []

for tgt in SINGLE_TARGETS:
    # Noted'interprétation
    # On entraîne une pipeline PAR cible en exploitant toutes les lignes où cette cible est connue.
    # Cela augmente drastiquement n (UTS≈738, Yield≈780, Charpy≈879, Elong≈700 au lieu de ~142 en intersection),

    mask = df[tgt].notna()
    n_rows = int(mask.sum())
    if n_rows < 50:
        print(f"[SKIP] {tgt}: seulement {n_rows} lignes non‑NaN (trop peu).")
        continue

    # X: on retire TOUTES les cibles pour éviter le data leakage (on suppose qu'on ne les aura pas en inference)
    X_t = df.loc[mask].drop(columns=SINGLE_TARGETS, errors="ignore").copy()
    y_t = df.loc[mask, tgt].astype(float).copy()

    print(f"\n--- {tgt} ---\nLignes disponibles: {n_rows}")

    # Split
    Xtr, Xte, ytr, yte = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

    # Détection des types sur l'apprentissage (pour éviter train/test shift de colonnes inconnues)
    num_features_t = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat_features_t = [c for c in Xtr.columns if c not in num_features_t]

    # Préprocesseurs (identiques à plus haut)
    num_transformer_t = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_transformer_t = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor_t = ColumnTransformer([
        ("num", num_transformer_t, num_features_t),
        ("cat", cat_transformer_t, cat_features_t),
    ])

    # Modèles
    rf_t = Pipeline([
        ("prep", preprocessor_t),
        ("model", RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=42)),
    ])
    et_t = Pipeline([
        ("prep", preprocessor_t),
        ("model", ExtraTreesRegressor(n_estimators=800, n_jobs=-1, random_state=42)),
    ])

    # CV plus stable
    # On utilise RepeatedKFold (5x3) pour réduire la variance des estimations CV
    rcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring_t = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    res_t = {}
    for name, pipe in {"RandomForest": rf_t, "ExtraTrees": et_t}.items():
        cv_res = cross_validate(pipe, Xtr, ytr, cv=rcv, scoring=scoring_t, n_jobs=-1, return_train_score=False)
        res_t[name] = {
            "MAE_cv_mean": -np.mean(cv_res["test_MAE"]),
            "RMSE_cv_mean": -np.mean(cv_res["test_RMSE"]),
            "R2_cv_mean": np.mean(cv_res["test_R2"]),
        }

    print("Validation croisée (moyenne 5x3 folds):")
    for k, v in res_t.items():
        print(f"{k:>12} | MAE={v['MAE_cv_mean']:.3f} | RMSE={v['RMSE_cv_mean']:.3f} | R2={v['R2_cv_mean']:.3f}")

    # Sélection du modèle : on choisit le meilleur au R carré moyen CV.
    # Interprétation observée:
    #  - UTS & Yield : ExtraTrees > RandomForest 
    #  - Elongation : parfois RF ≈ ET (plus de bruit)
    #  - Charpy : beaucoup d'information portée par Charpy_temp et traitements thermiques. gGains nets en données par‑cible.

    best_name_t = max(res_t.keys(), key=lambda n: res_t[n]["R2_cv_mean"])
    best_pipe_t = rf_t if best_name_t == "RandomForest" else et_t
    print(f"Meilleur modèle pour {tgt}: {best_name_t}")

    # Fit final et hold‑out
    best_pipe_t.fit(Xtr, ytr)
    pred_t = best_pipe_t.predict(Xte)

    mae_t = mean_absolute_error(yte, pred_t)
    mse_t = mean_squared_error(yte, pred_t)
    rmse_t = np.sqrt(mse_t)
    r2_t = r2_score(yte, pred_t)

    print(f"Hold‑out {tgt} | MAE={mae_t:.3f} | RMSE={rmse_t:.3f} | R2={r2_t:.3f}")

    # Sauvegarde des prédictions
    out_df = pd.DataFrame({f"y_true_{tgt}": yte.values, f"y_pred_{tgt}": pred_t}, index=yte.index)
    out_path = f"./welddb_predictions_{tgt}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Prédictions {tgt} sauvegardées dans {out_path}")

    per_target_reports.append({
        "target": tgt,
        "n_rows": n_rows,
        "cv_R2": res_t[best_name_t]["R2_cv_mean"],
        "test_R2": r2_t,
        "test_RMSE": rmse_t,
        "best_model": best_name_t,
    })

# Résumé
if per_target_reports:
    print("\n=== Récapitulatif par cible (meilleur modèle CV puis hold‑out) ===")
    for r in per_target_reports:
        print(f"{r['target']:>16} | rows={r['n_rows']:4d} | CV R2={r['cv_R2']:.3f} | Test R2={r['test_R2']:.3f} | Test RMSE={r['test_RMSE']:.3f} | {r['best_model']}")