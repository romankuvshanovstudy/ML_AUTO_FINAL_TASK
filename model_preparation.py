import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    train_data = pd.read_csv("train/temperature_train_scaled.csv")

    X_train = train_data[['Day']]
    y_train = train_data['Temperature']

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'temperature_model.pkl')

if __name__ == "__main__":
    train_model()
