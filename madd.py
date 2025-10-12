import pandas as pd
df = pd.read_excel("D:\\ai project\\global-uk-tariffv1\\global-uk-tariff - Copy (2).xlsx")
print(df[df['commodity'].str.startswith('04')])