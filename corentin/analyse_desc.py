# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
print(df_raw.head(3))

# %%
miss = df_raw.isna().mean().sort_values(ascending=False)
miss_df = miss.to_frame(name="missing_ratio").reset_index().rename(columns={"index": "column"})
print(miss_df)


# %%
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
desc = df_raw[num_cols].describe().T
print(desc)


# %%
# --- Corrélations avec les variables de qualité ---
target_vars = ["CharpyJ", "Elongation_pct", "UTS_MPa"]

# On garde uniquement les colonnes numériques
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

# Calcul des corrélations de Pearson
corrs = df_raw[num_cols].corr(method='pearson')[target_vars]

# Tri des corrélations en fonction de la quantité de missing values
corrs_sorted = corrs.reindex(miss_df['column'])


print("=== Corrélation absolue avec les variables de qualité ===")
print(corrs_sorted.round(3))

# %%
# --- Visualisation heatmap Plotly ---
fig = go.Figure(
    data=go.Heatmap(
        z=corrs_sorted.values,
        x=corrs_sorted.columns,
        y=corrs_sorted.index,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="corrélation")
    )
)

fig.update_layout(
    title="Corrélations entre les variables numériques et les indicateurs de qualité",
    xaxis_title="Variables cibles (qualité)",
    yaxis_title="Variables explicatives"
)
fig.show()

# %%
print(corrs_sorted.round(3))