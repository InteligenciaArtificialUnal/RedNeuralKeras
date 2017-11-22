import numpy as np
import pandas as pd

def test():
    dataset_path = "C:\\Users\\Daniel\\Desktop\\doc1.csv"
    data = pd.read_csv(dataset_path, header=0, error_bad_lines=False)
    print (data)


test()