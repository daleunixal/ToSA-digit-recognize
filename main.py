import pandas as pd
from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.utils import np_utils
# Если ругается Tensorflow проверьте версию PyEnv should be 3.X:| in [7-9]
from tensorflow.python.keras.models import Sequential

# разбиваем данные на RAW и поддержку
(x_train_data, y_train_data), (x_valid_data, y_valid_data) = mnist.load_data()

# Решейпим данные под входную модель для CNN
x_train_data = x_train_data.reshape(60000, 28 * 28).astype('float32')
x_valid_data = x_valid_data.reshape(10000, 28 * 28).astype('float32')

# Нормализируем данные как 8-битные
x_train_data = x_train_data / 255
x_valid_data = x_valid_data / 255

# Разбираем лейблы RAW и поддержки на категоризированный вектор (Пример => 5 -> [0,0,0,0,0,1,0,0,0,0])
train_label = np_utils.to_categorical(y_train_data)
valid_label = np_utils.to_categorical(y_valid_data)

# Определяем функцию создания модели
def create_model():
    model = Sequential()
    model.add(Input(shape=(28 * 28)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Собираем скомпилированную модель и отдаем ей RAW/support данные для обучения
model = create_model()
model.fit(x_train_data, train_label, epochs=15, batch_size=128, verbose=1, validation_split=0.1)
model.evaluate(x_valid_data, valid_label, verbose=1)


# Сохраняем модель для дальнейшей работы с фронтом
model.save("model.h5")
print("Веса сохранены в ./model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Модель CNN сохранена в ./model.json")
