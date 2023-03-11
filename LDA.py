from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import pandas as pd



def read_data(path):
    data = pd.read_excel(path, header=None)
    raw_data = data.to_numpy()[0:-2, 1:].T
    classes = data.to_numpy()[-1, 1:]
    classes = list(classes)
    return raw_data, classes


def select_data(labled_data):
    pass


def LDA(data, target,test_data):
    mylda = lda()
    mylda.fit(data, target)
    w = mylda.scalings_
    transformed_data = mylda.transform(data)
    prediction=mylda.predict(test_data)
    return transformed_data, w,prediction


if __name__ == "__main__":
    data_array, label = read_data("lda-train.xls")
    test,test_label=read_data("lda-test.xls")

    result, scaling,predict = LDA(data_array, label,test)

    result=pd.DataFrame(result)
    label=pd.DataFrame(label)
    result['label']=label.reset_index(drop=True)
    result.to_csv("LDA.csv")

    predict=pd.DataFrame(predict)
    test_label=pd.DataFrame(test_label)
    test=pd.DataFrame(test)
    test['label']=test_label.reset_index(drop=True)
    test['predicted_label']=predict.reset_index(drop=True)
    test.T.to_csv("predicted.csv")

    pd.DataFrame(scaling).to_csv("LDA-scaling.csv")
