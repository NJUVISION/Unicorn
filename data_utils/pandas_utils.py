
import pandas as pd
import numpy as np

def mean_dataframe(df_list):
    """calculate the average value of input df_list
    """
    df_mean = pd.DataFrame()
    for col in df_list[0].columns:
        try: 
            df_mean[col] = np.stack([df[col] for df in df_list]).mean(axis=0)
        except TypeError:
            continue

    return df_mean