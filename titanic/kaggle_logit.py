# modified from LinkanRay's python code
# https://www.kaggle.com/linkanray/titanic/titanic-logistic-regression
# changes:
# - code format
# - add training performance

import numpy as np
import pandas as pd
# from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import accuracy_score


def load_file_train(data_dir, X_cols):
    df = pd.read_csv(data_dir + "train.csv")

    # change male to 1 and female to 0
    df["Sex"] = df["Sex"].\
                      apply(lambda sex: 1 if sex == "male" else 0)
    # handle missing values of age
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    survived = df["Survived"].values
    data = df[X_cols].values
    return survived, data


def load_file_test(data_dir, X_cols):
    df = pd.read_csv(data_dir + "test.csv")

    # change male to 1 and female to 0
    df["Sex"] = df["Sex"].\
                      apply(lambda sex: 1 if sex == "male" else 0)
    # handle missing values of age
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    data = df[X_cols].values
    pass_id = df["PassengerId"].values
    return data, pass_id


def learn_model(survived, data_train, data_test, pass_id, output_dir):
    model = LogisticRegression()
    model.fit(data_train, survived)

    # evaluate performance
    print("training performance:")
    train_pred = model.predict(X=data_train)
    print(pd.crosstab(train_pred, survived))
    print(model.score(data_train, survived))

    # predict test data
    print("generate learning results regarding test")
    predicted = model.predict(data_test)
    output = pd.DataFrame(columns=["PassengerId", "Survived"])
    output["PassengerId"] = pass_id
    output["Survived"] = predicted.astype(int)
    output.to_csv(output_dir + "kaggle_logit.csv", index=False)

    print("finished")


def main():
    # parameters:
    data_dir = "data/"
    output_dir = "output/"
    # what features to used should be determined beforehand
    X_cols = ["Pclass", "Sex", "Age"]

    survived, data_train = load_file_train(data_dir, X_cols)
    data_test, pass_id = load_file_test(data_dir, X_cols)

    learn_model(survived, data_train, data_test, pass_id, output_dir)

if __name__ == "__main__":
    main()
