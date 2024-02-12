import numpy as np

from scipy.stats import zscore
from IPython.display import display


def z_score_outliers(df, col, threshold):
    z_scores = zscore(df[[f'{col}']])
    outliers = df[abs(z_scores) > threshold]
    
    print('\n', f'Possible outliers for "{col}" identified with Z-Score method')
    display(outliers) 


def iqr_outliers(df, col, ind):
    Q1 = np.percentile(df[[f'{col}']], 25)
    Q3 = np.percentile(df[[f'{col}']], 75)
    IQR = Q3 - Q1
    threshold = ind * IQR
    outliers = df[(df[f'{col}'] < Q1 - threshold) | (df[f'{col}'] > Q3 + threshold)]

    print('\n', f'Possible outliers for "{col}" identified with IQR method')
    display(outliers)