
import pandas as pd
"""
读取信息, 数据处理
"""
CSV_PATH = "F:\Dataset\Dogs\labels.csv"
TRAIN_PATH = "F:\Dataset\Dogs\\train.zip"
TEST_PATH = "F:\Dataset\Dogs\\test.zip"

df = pd.read_csv(CSV_PATH)

breeds = list(df['breed'])
img_id = df['id']

idx2label = []
for breed in set(breeds):
    idx2label.append(breed)
label2idx = {label: idx for idx, label in enumerate(idx2label)}

