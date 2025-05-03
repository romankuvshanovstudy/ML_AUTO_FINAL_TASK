import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    """
    Trains a linear regression model to predict temperature based on the day.

    The function reads preprocessed and scaled training data from 
    'train/temperature_train_scaled.csv', where it uses the 'Day' column 
    as the input feature and the 'Temperature' column as the target variable.
    It fits a LinearRegression model from scikit-learn and saves the trained 
    model to a file named 'temperature_model.pkl' using joblib for later use.
    """
    train_data = pd.read_csv("train/temperature_train_scaled.csv")

    X_train = train_data[['Day']]
    y_train = train_data['Temperature']

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'temperature_model.pkl')

if __name__ == "__main__":
    train_model()
