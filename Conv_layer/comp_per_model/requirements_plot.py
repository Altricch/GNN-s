import json
import pandas as pd
import matplotlib.pyplot as plt


# This file is used to plot the average CPU, Memory and Time taken by different models with different number of convolution layers
# The data is taken from the file compare_all.py_requirements.json

filename = 'compare_all.py_requirements.json'

with open(filename, 'r') as file:
    data = json.load(file)


df = pd.DataFrame.from_dict({k: v for k, v in data.items()}, orient='index')

df.reset_index(inplace=True)
df.columns = ['Configuration'] + list(df.columns[1:])


# Splitting Configuration into separate columns
df[['Conv', 'LR', 'Hidden', 'Model']] = df['Configuration'].str.split(', ', expand=True)

# remove unnecessary characters
df[['Conv1', 'Conv2', 'Conv']] = df['Conv'].str.split(' ', expand=True)
df[['LR1', 'LR']] = df['LR'].str.split(' ', expand=True)
df[['Hidden1', 'Hidden']] = df['Hidden'].str.split(' ', expand=True)
df[['Model1', 'Model']] = df['Model'].str.split(' ', expand=True)


df = df[['Conv', 'LR', 'Hidden', 'Model', 'CPU', 'Memory', 'Time']]
df.drop(columns=['LR'], inplace=True)

df[['Conv', 'Hidden', 'CPU', 'Memory', 'Time']] = df[['Conv', 'Hidden', 'CPU', 'Memory', 'Time']].apply(pd.to_numeric)

grouped_df = df.groupby(['Conv', 'Model']).mean().reset_index()
grouped_df.drop(columns=['Hidden'], inplace=True)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

columns = ['CPU', 'Memory', 'Time']

for column in columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in grouped_df['Model'].unique():
        model_df = grouped_df[grouped_df['Model'] == model]
        ax.plot(model_df['Conv'], model_df[column], label=model)

    ax.set_xlabel('Number of Convolution Layers')
    ax.set_ylabel(column)
    ax.set_title(f'{column} vs Number of Convolution Layers')
    ax.legend()

    # Save the figure
    fig_name = f'{column}_vs_Convolutions.png'
    fig.savefig(fig_name)
    plt.close(fig) 
