import pandas as pd
import matplotlib
import asyncio
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import plotly.express as px



def get_columns():
    df = pd.read_csv("skylearn/clean/clean.csv")
    return df.columns

def pair_plot():
    df = pd.read_csv("skylearn/clean/clean.csv")

    try:
        sns_plot = sns.pairplot(df, height=2.5)
        sns_plot.savefig("skylearn/static/img/pairplot1.png")
    except:
        sns_plot = px.scatter_matrix(df)
        sns_plot.write_image("skylearn/static/img/pairplot1.png")

    return True


def xy_plot(col1, col2):
    df = pd.read_csv("skylearn/clean/clean.csv")
    return df

def hist_plot(df,feature_x):
    # df=df.sort_values([feature_x], axis=0, ascending=True, inplace=True) 
    x= df[feature_x]
    x.to_csv("skylearn/visualization/col.csv",mode="w", index=False,header=['price'])
    with open("skylearn/visualization/col.csv", 'r') as filehandle:
        lines = filehandle.readlines()
        lines[-1]=lines[-1].strip()
    with open("skylearn/visualization/col.csv", 'w') as csvfile:
        for i in lines:
            csvfile.write(i)  
    return True
