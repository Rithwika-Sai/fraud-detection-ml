import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model():

    data = pd.read_csv("dataset_sample.csv")

    X = data.drop("is_fraud", axis=1)
    y = data["is_fraud"]

    model = RandomForestClassifier()
    model.fit(X,y)

    return model
