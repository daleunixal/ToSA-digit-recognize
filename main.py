import numpy as np
import pandas as pd
from keras import backend as k
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
# Если ругается Tensorflow проверьте версию PyEnv should be 3.X:| in [7-9]

data = pd.read_csv("./MNIST/mnist_train.csv")
data = data.values
# Забираем лейблы.
label = data[:, 0]

# Подгружаем данные
data = data[:, 1:]

# разбиваем данные на RAW и поддержку
train_data = data[:35000, :]
valid_data = data[35000:, :]

# Решейпим данные под входную модель для CNN
train_data = train_data.reshape(train_data.shape[0], 1, 28, 28).astype('float32')
valid_data = valid_data.reshape(valid_data.shape[0], 1, 28, 28).astype('float32')

# Нормализируем данные как 8-битные
train_data = train_data / 255
valid_data = valid_data / 255

# разбиваем лейблы на RAW и поддержку
train_label = label[:35000]
valid_label = label[35000:]


# Разбираем лейблы RAW и поддержки на категоризированный вектор (Пример => 5 -> [0,0,0,0,0,1,0,0,0,0])
train_label = np_utils.to_categorical(train_label)
valid_label = np_utils.to_categorical(valid_label)

# Устанавливаем для модели Keras формат входных данных изображения
k.set_image_data_format('channels_first')

# Устанавливаем сид(https://en.wikipedia.org/wiki/Random_seed) для воспроизведения модели
seed = 7
np.random.seed(seed)


# Определяем функцию создания модели
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Собираем скомпилированную модель и отдаем ей RAW/support данные для обучения
model = create_model()
model.fit(train_data, train_label, validation_data=(valid_data, valid_label), epochs=15, batch_size=200, verbose=2)


# Сохраняем модель для дальнейшей работы с фронтом
model.save("model.h5")
print("Веса сохранены в ./model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Модель CNN сохранена в ./model.json")
