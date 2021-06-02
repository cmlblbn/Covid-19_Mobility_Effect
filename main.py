

import data_train
import data_preprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn





if __name__ == '__main__':

    df,subregion_scores = data_preprocess.data_preprocess()
    subregion_scores = data_train.xgbooster(df,subregion_scores)
    for data in subregion_scores:
        print(data, " prediction boost score is: %" , subregion_scores[data])
