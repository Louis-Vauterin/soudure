# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline



# %%
DATA_PATH = "welddb/welddb.data"
COLS = [
    "C_wt_pct", "Si_wt_pct", "Mn_wt_pct", "S_wt_pct", "P_wt_pct", "Ni_wt_pct", "Cr_wt_pct",
    "Mo_wt_pct", "V_wt_pct", "Cu_wt_pct", "Co_wt_pct", "W_wt_pct",
    "O_ppm", "Ti_ppm", "N_ppm", "Al_ppm", "B_ppm", "Nb_ppm", "Sn_ppm", "As_ppm", "Sb_ppm",
    "Current_A", "Voltage_V", "AC_or_DC", "Electrode_polarity",
    "HeatInput_kJ_per_mm", "InterpassTemp_C", "WeldType",
    "PWHT_Temp_C", "PWHT_Time_h",
    "YieldStrength_MPa", "UTS_MPa", "Elongation_pct", "ReductionArea_pct",
    "CharpyTemp_C", "CharpyJ", "Hardness_kg_per_mm2",
    "FATT_50pct",
    "PrimaryFerrite_pct", "FerriteSecondPhase_pct", "AcicularFerrite_pct", "Martensite_pct", "FerriteCarbideAgg_pct",
    "WeldID"
]



# %%
df_raw = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLS,
        sep=r"\s+",
        na_values=["N", "n"],
        engine="python"
    )
# %%
miss = df_raw.isna().mean().sort_values(ascending=False)
miss_df = miss.to_frame(name="missing_ratio").reset_index().rename(columns={"index": "column"})
miss_df


# %% [markdown]
# J'ai trouvé dans la littérature que les variables les plus à même d'expliquer la qualité d'une soudure sont le charpyJ, UTS et Elongation

# %%
target_vars = ["CharpyJ", "Elongation_pct", "UTS_MPa"]

num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

corrs = df_raw[num_cols].corr(method='pearson')[target_vars]

# Tri des corrélations en fonction de la quantité de missing values
corrs_sorted = corrs.reindex(miss_df['column'])


print("=== Corrélation absolue avec les variables de qualité ===")
print(corrs_sorted.round(3))
fig = px.imshow(
    corrs_sorted.abs(),
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Correlation absolue entre les variables numériques et les variables de qualité"
)
fig.show()

# %% [markdown]
# On va donc retirer toutes les variables qui on un taux de missing value élevé ainsi qu'une corrélation faible ou inexistante avec les variables explicatives.

# %%
COLS_TO_DROP = ["FATT_50pct", "W_wt_pct", "FerriteCarbideAgg_pct", "Martensite_pct", "FerriteSecondPhase_pct", "AcicularFerrite_pct", "PrimaryFerrite_pct", "Co_wt_pct", "Hardness_kg_per_mm2", "As_ppm", "Sb_ppm", "Sn_ppm", "B_ppm", "Cu_wt_pct"]
df = df_raw.drop(columns=COLS_TO_DROP)

# %%
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
desc = df[num_cols].describe().T
desc

# %%
row_missing_ratio = df.isna().mean(axis=1)
print(max(row_missing_ratio))
fig = px.histogram(row_missing_ratio, nbins=12,
                   title="Distribution du pourcentage de valeurs manquantes par ligne",
                   labels={"value": "Taux de valeurs manquantes"})
fig.show()

# %% [markdown]
# Toutes les lignes ont au moins la moitié de leurs valeurs. On va les conserver

# %% [markdown]
# Maintenant, on va compléter les lignes qui contiennent des NaN

# %%
# the paper describes how to handle chemicals : 

impurity_elements = ["P_wt_pct", "S_wt_pct"]
deliberate_elements = ["Mn_wt_pct", "Ni_wt_pct", 
                       #"Co_wt_pct", 
                       "Cr_wt_pct", "Mo_wt_pct"]

for col in impurity_elements + deliberate_elements:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# For P and S : fill with mean
for col in impurity_elements:
    df[col] = df[col].fillna(df[col].mean())

# Fill deliberate alloying elements (Mn, Ni, etc.) with 0
for col in deliberate_elements:
    df[col] = df[col].fillna(0)

# %%
remaining_missing = df.isna().mean().sort_values(ascending=False)
print(remaining_missing[remaining_missing > 0])

# %%
#uts before imputation :
px.histogram(df, x="UTS_MPa", nbins=30, title="UTS before imputation").show()

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

knn = KNNImputer(n_neighbors=5)
df[num_cols] = knn.fit_transform(df[num_cols])

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=5))
])
df[num_cols] = pipeline.fit_transform(df[num_cols])

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df.isna().sum().sum())
px.histogram(df, x="UTS_MPa", nbins=30, title="UTS after imputation").show()


