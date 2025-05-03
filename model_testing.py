import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def test_model():
    test_data = pd.read_csv("test/temperature_test_scaled.csv")
    model = joblib.load('temperature_model.pkl')

    X_test = test_data[['Day']]
    y_test = test_data['Temperature']
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    test_model()
