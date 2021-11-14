import cv2
import numpy as np
from matplotlib import pyplot as plt


def model():
    from keras.models import model_from_json
    model_file_stream = open('model.json', 'r')
    loaded_model_json = model_file_stream.read()
    model_file_stream.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("[UNIXAL_LILITH]: CNN Model is ready")

    return loaded_model


def input_prepare(img):
    img = np.asarray(img)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    plt.imshow(img)
    img = img / 255
    img = img.reshape(1, 784)
    return img



def main():
    loaded_model = model()
    img_origin = cv2.imread('1.png', 3)
    img = img_origin.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = input_prepare(img)

    pred = loaded_model.predict(img)
    plt.imshow(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))
    plt.title(np.argmax(pred, axis=1))
    plt.show()


main()
