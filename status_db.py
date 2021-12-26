import pandas as pd
from collections import Counter 
from pprint import pprint

data = pd.read_pickle("labels_final.pkl")
pprint(Counter(data))


