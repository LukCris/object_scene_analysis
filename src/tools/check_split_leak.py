# check_split_leak.py (snippet veloce)
import pandas as pd
from pathlib import Path

def read_paths(csv):
    df = pd.read_csv(csv)
    return set(df['path'].astype(str).str.replace('\\','/', regex=False))

train = read_paths('../../manifests/train.csv')
valid = read_paths('../../manifests/valid.csv')
test  = read_paths('../../manifests/test.csv')

print('train ∩ valid:', len(train & valid))
print('train ∩ test :', len(train & test))
print('valid ∩ test :', len(valid & test))
