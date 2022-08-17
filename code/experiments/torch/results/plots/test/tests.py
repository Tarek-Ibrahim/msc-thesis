
# import numpy as np
from scipy import stats
import pandas as pd

eps=100 #50
xp_name="xp2_full"

activedr_col="PPO + active_dr (map_delta / sac)"
maml_activedr_col="MAML + active_dr (map_delta / sac)"
meta_autodr_col="MAML + auto_dr"

df = pd.read_pickle(f"lowest_values_{eps}_episodes_lunarlander_{xp_name}.pkl")

res_maml = stats.mannwhitneyu(df[activedr_col],df[maml_activedr_col])

res_autodr = stats.mannwhitneyu(df[activedr_col],df[meta_autodr_col])



