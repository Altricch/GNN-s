import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True, precision=4)

# This File is responsible for loading the ADGN_Message.py_requirements.json file in order to the plot its contents

# Importing the file
data = {}
file_path = "ADGN_Message.py_requirements.json"
if os.path.exists(file_path):
    # Open the file and load the data
    with open(file_path, "r") as file:
        data = json.load(file)
else:
    raise Exception("File not found")

# Extracting and formatting its contents
configs = []
values = []

for el in data:
    temp = str(el).replace(",", "").split(" ")
    idx_con_lr_hid = [int(temp[0]), float(temp[2]), float(temp[4]), float(temp[-1])]
    configs.append(np.array(idx_con_lr_hid))

    temp = str(data[el]).replace(",", ":").split(":")
    cpu_mem_time = [float(temp[1]), float(temp[3]), float(temp[-1][:-1])]
    values.append(np.array(cpu_mem_time))

configs = np.array(configs)
values = np.array(values)
configs = np.flip(configs[:, 1:], 1)  # Better ordering for plotting


# Dataframe creation and plotting

# EXECUTION TIME
df_data_time = np.concatenate([configs, values[:, -1:]], axis=1).astype(float)
columns = [
    "Hidden Dimension",
    "Learning Rate",
    "Convolutional Layers",
    "Execution Time",
]
df_time = pd.DataFrame(data=df_data_time, columns=columns)
print(df_time)

fig = px.parallel_coordinates(
    df_time,
    color="Execution Time",
    dimensions=columns,
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    title="Grid-Search Execution Time (in seconds)",
    height=500,
    width=1000,
)
fig.show()


# CPU USAGE
df_data_cpu = np.concatenate([configs, values[:, :1]], axis=1).astype(float)
columns = ["Hidden Dimension", "Learning Rate", "Convolutional Layers", "CPU Usage"]
df_time = pd.DataFrame(data=df_data_cpu, columns=columns)
print(df_time)

fig = px.parallel_coordinates(
    df_time,
    color="CPU Usage",
    dimensions=columns,
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    title="Grid-Search CPU Usage (in percentage)",
    height=500,
    width=1000,
)

fig.show()

# MEMORY USAGE
df_data_mem = np.concatenate([configs, values[:, 1:2]], axis=1).astype(float)
columns = ["Hidden Dimension", "Learning Rate", "Convolutional Layers", "Memory Usage"]
df_time = pd.DataFrame(data=df_data_mem, columns=columns)
print(df_time)

fig = px.parallel_coordinates(
    df_time,
    color="Memory Usage",
    dimensions=columns,
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    title="Grid-Search Memory Usage (in percentage of total memory)",
    height=500,
    width=1000,
)

fig.show()
