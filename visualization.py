from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def cumulative_percentage_variance(data, xtitle, ytitle):
    fig = px.bar(data,
                 x=xtitle,
                 y=ytitle,
                 color=xtitle)
    fig.show()


def scatter_plot(df, x, y, color):
    fig = px.scatter(df, x=x, y=y, color=color)
    fig.show()


def cumulative_var():
    data = pd.read_excel("eig.xlsx", header=None)
    data = pd.DataFrame(np.array([data[0], data[3]]).T)
    data.columns = ['pc', 'cumulative variance percentage']
    cumulative_percentage_variance(data, 'pc', 'cumulative variance percentage')


def score_plot():
    score = pd.read_csv("score.csv")
    label = score['label']
    data_frame = pd.DataFrame([score['0'], score['1'], label]).T
    data_frame.columns = ['PC1', 'PC2', 'label']
    scatter_plot(data_frame, 'PC1', 'PC2', 'label')


def loading_plot():
    loading = pd.read_csv("loading.csv")
    label = loading.columns
    data_frame = pd.DataFrame([loading.iloc[1][1:], loading.iloc[2][1:], label]).T
    data_frame.columns = ['PC1', 'PC2', 'label']
    scatter_plot(data_frame, 'PC1', 'PC2', 'PC1')


def loading_plot_matplt():
    loading = pd.read_csv("loading.csv")
    label = loading.columns
    data_frame = pd.DataFrame([loading.iloc[1][1:], loading.iloc[2][1:], label]).T
    data_frame.columns = ['PC1', 'PC2', 'label']
    for i in range(0, len(data_frame)):
        plt.arrow(0, 0, data_frame.iloc[i][0],
                  data_frame.iloc[i][1],
                  color='b')
        """plt.text(data_frame.iloc[i][0],
                 data_frame.iloc[i][1],
                 label[i], color='r')"""

    # plt.show()
    plt.savefig("loadingplot.png")


def lda():
    data = pd.read_csv("LDA.csv")
    data['y'] = [0] * len(data)
    data.columns = ['index', 'samples', 'label', 'y']
    scatter_plot(data, 'samples', 'y', 'label')
