import pandas as pd
import numpy as np

def mean_file(file_list):
    if len(file_list)==1:
        return pd.read_csv(file_list[0])

    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df_list.append(df)
        
    df = mean_dataframe(df_list)
    
    return df


def mean_dataframe(df_list):
    """calculate the average value of input df_list
    """
    df_mean = pd.DataFrame()
    # print([df_list[i].columns for i in range(len(df_list))])
    for col in df_list[0].columns:
        # if col in ['mae', 'mse', 'RGB-PSNR', 'mped', 'graphsim']: continue
        try: 
            df_mean[col] = np.stack([df[col] for df in df_list]).mean(axis=0)
        except TypeError:
            continue

    return df_mean

