import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def test_model():
    """
    Тестирует предварительно обученную модель на масштабированных тестовых данных и вычисляет среднеквадратичную ошибку.

    Загружает масштабированные тестовые данные и предварительно обученную модель, выполняет предсказания,
    используя столбец 'Day' в качестве входных данных, и вычисляет среднеквадратичную ошибку между
    предсказанными и фактическими значениями столбца 'Temperature'.

    Входные файлы:
        - test/temperature_test_scaled.csv: Масштабированные тестовые данные со столбцами 'Day' и 'Temperature'.
        - temperature_model.pkl: Файл с предварительно обученной моделью.

    Вывод:
        - Выводит среднеквадратичную ошибку предсказаний модели.
    """
    test_data = pd.read_csv("test/temperature_test_scaled.csv")
    model = joblib.load('temperature_model.pkl')

    X_test = test_data[['Day']]
    y_test = test_data['Temperature']
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Среднеквадратичная ошибка: {mse}')

if __name__ == "__main__":
    test_model()
