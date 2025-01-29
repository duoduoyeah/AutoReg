from linearmodels.datasets import jobtraining
from linearmodels import PanelOLS

data = jobtraining.load()
print(data.head())
input("Press any key to continue...")


orig_mi_data = data.set_index(["fcode", "year"])
# Subset to the relevant columns and drop missing to avoid warnings
mi_data = orig_mi_data[["lscrap", "hrsemp"]]
mi_data = mi_data.dropna(axis=0, how="any")

print(mi_data.head())
input("Press any key to continue...")


mod = PanelOLS(mi_data.lscrap, mi_data.hrsemp, entity_effects=True)
print(mod.fit())
