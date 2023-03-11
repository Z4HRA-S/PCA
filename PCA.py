from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import numpy as np
import pandas as pd


def read_xls(path):
    df = pd.read_excel(path, header=None)
    raw_data=df.to_numpy().T
    features=df[0][:-2]
    labels=df.iloc[-2][1:]
    return raw_data,features,labels


def centering(data_matrix):
    """
    we assume variables are in columns and samples are in rows
    so each row is a data_point
    :param data_matrix:
    :return: centered data by mean for each column
    """
    column_mean = np.mean(data_matrix, axis=0)
    return data_matrix - column_mean


def pca_by_svd(A):
    """

    :param A:
    :return: score matrix and loading matrix
    """
    A = np.array(A, dtype='float64')
    AAt = A.dot(A.T)
    eigen_vals, u = np.linalg.eig(AAt)
    idx = np.argsort(eigen_vals)[::-1]
    s = eigen_vals[idx]  # sort eigenvalues descending
    u = u[:, idx]  # sort corresponding eigenvector

    i = 1
    while (sum(s[:i]) / sum(s)) < 0.95:
        i += 1
    print(i)

    s = np.diag(np.sqrt(s[:i]))  # select i first eigenvalues
    u = np.array(u[:, 0:i], dtype='float64')  # select i first eigenvectors

    score_matrix = u.dot(s)
    loading_matrix = np.linalg.inv(s).dot((u.T).dot(A))
    return score_matrix, loading_matrix


if __name__ == "__main__":
    path = input()
    raw_data,features,label = read_xls(path)
    centered_data = centering(raw_data[1:, 0:-2])
    score, loading = pca_by_svd(centered_data)
    score=pd.DataFrame(score)
    score['label']=label.reset_index(drop=True)
    score.to_csv("score.csv")
    loading = pd.DataFrame(loading)
    loading.columns=features
    loading.to_csv("loading.csv")
