import pickle

with open("D:\\ai project\\global-uk-tariffv1\\tariff_model.pkl", "rb") as f:
    bundle = pickle.load(f)
print(f"Type: {type(bundle)}")
print(f"Keys: {bundle.keys() if isinstance(bundle, dict) else dir(bundle)}")
print(f"Version: {bundle['version']}, Records: {len(bundle['df_records'])}")