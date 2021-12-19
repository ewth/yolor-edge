import os
from glob import glob
import pandas as pd


# Grab most recently updated files
files = list(filter(os.path.isfile, glob('./csv/*.csv')))
files.sort(key=lambda x: os.path.getmtime(x))
file = files[0]

data = pd.read_csv(file)
# print(data.head())

# max ap: 
sorted_by_map = data.sort_values(by='ap', axis=0, ascending=False)

# id,model,batch_size,image_size,ap,ap_50,ap_75,ap_S,ap_M,ap_L
labels = 'id,model,batch_size,image_size,ap,ap_50,ap_75,ap_S,ap_M,ap_L'.split(',')
print(f"{(")
print("Max AP:")
# print(f"{(for x in labels ")
# max_ap = sorted_by_map['ap'].iloc[0]
# max_ap_index = max_ap['.index[0]']
