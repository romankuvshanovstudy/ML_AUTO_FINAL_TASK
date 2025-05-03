#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Создаем виртуальное окружение..."
    python3 -m venv venv
else
    echo "Виртуальное окружение уже существует."
fi

echo "Активируем виртуальное окружение..."
source venv/bin/activate

echo "Устанавливаем зависимости..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Запускаем генерацию данных..."
python3 data_creation.py

echo "Запускаем предобработку данных..."
python3 model_preprocessing.py

echo "Запускаем обучение модели..."
python3 model_preparation.py

echo "Запускаем тестирование модели..."
python3 model_testing.py

