from pathlib import Path

import pandas as pd

raw_dpath = Path(__file__).resolve().parents[1] / "datasets" / "DatabaseExtract_v_1.01.csv"

data_tf = pd.read_csv(raw_dpath)

print(data_tf.shape)

print(data_tf.columns)

