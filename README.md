В ходе обучения использовались 
Python 3.11
CUDA 13.0
PyTorch 2.11.0

для установки необходимых библиотек выполните команду:
pip install -r requirements.txt


Также стоит отдельно установить сначала PyTorch, а затем остальные библиотеки из requirements.txt, так как в файле указаны версии, которые могут не совпадать с последними версиями PyTorch и CUDA.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

Затем необходимо скачать файл модели DLIB для предсказания точек
Структура
project/

├── utils.py               # Утилитарные функции для обучения

├── Dataset.py               # Класс датасета

├── models.py                # Архитектуры моделей

├── main.py                 # Обучение модели

├── Preproces_mult.py        # препроцессинг данных

├── test.py                  # Тестирование и построение CED-графиков

├── requirements.txt         # Зависимости проекта

├── README.md                # Документация



├── meta_train_300w.json
├── meta_test_300w.json
├── meta_train_menpo.json
├──meta_test_menpo.json

├── data/

│   ├── 300W/

│   │   ├── train/

│   │   └── test/

│   │

│   └── Menpo/

│       ├── train/

│       └── test/

├── runs/                    # TensorBoard логи

├── models/                  # Сохранённые веса моделей

После установки датасетов и модели можно запустить код 

python Preproces_mult.py # препроцессинг данных нужно запустить 1 раз чтобы получить Json с ббоксами
python main.py  # запуск обучения, сейчас при запуске будут обучаться все модели
python test.py  # запуск тестирования и построения CED-графиков

