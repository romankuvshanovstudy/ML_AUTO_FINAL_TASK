import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data():
    train_data = pd.read_csv("train/temperature_train.csv")
    test_data = pd.read_csv("test/temperature_test.csv")

    scaler = StandardScaler()

    train_data['Temperature'] = scaler.fit_transform(train_data[['Temperature']])
    test_data['Temperature'] = scaler.transform(test_data[['Temperature']])

    train_data.to_csv("train/temperature_train_scaled.csv", index=False)
    test_data.to_csv("test/temperature_test_scaled.csv", index=False)

if __name__ == "__main__":
    preprocess_data()
