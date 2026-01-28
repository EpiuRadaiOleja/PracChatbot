import pandas as pd
from tqdm import tqdm



meta = pd.read_excel('./source_pdfs/src_pdfs.xlsx')
print(meta.head())

for t in tqdm(range(10)):
    print("Cows")
