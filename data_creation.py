import numpy as np
import os
import pandas as pd

def generate_data():
    days = np.arange(1, 366)
    temperatures = 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 2, len(days))

    anomalies = np.random.choice([False, True], size=len(days), p=[0.98, 0.02])
    temperatures[anomalies] = temperatures[anomalies] + np.random.choice([10, -10], size=anomalies.sum())

    split_idx = int(0.8 * len(days))
    train_data = pd.DataFrame({'Day': days[:split_idx], 'Temperature': temperatures[:split_idx]})
    test_data = pd.DataFrame({'Day': days[split_idx:], 'Temperature': temperatures[split_idx:]})

    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    train_data.to_csv("train/temperature_train.csv", index=False)
    test_data.to_csv("test/temperature_test.csv", index=False)

if __name__ == "__main__":
    generate_data()
