import pandas as pd
import zipfile
import gdown
import os

def download_and_prepare():

    if not os.path.exists('train.zip'):
        gdown.download(id='1TMugQbE9EoapzN2rLlOjvEHO5ArviwrL', output='train.zip', quiet=False)

    with zipfile.ZipFile('train.zip', 'r') as zip_ref:
        zip_ref.extractall(".")

    final = pd.read_csv("train_bi_2.1.csv")
    final = final[:50]
    return final

if __name__ == "__main__":
    df = download_and_prepare()
    print(df.head())
