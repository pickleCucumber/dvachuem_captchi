import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# инициализация
data = []
labels = []

# цикл по входящим изображениям
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # загружаем изображение и конвертируем в серое
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, 25, 255, L2gradient=False)
    #cv2.imshow(image, cmap='gray')
    # изменение размера буквы, чтобы она поместилась в поле 20Х20 пикселей
   # image = resize_to_fit(image, 20, 20)

    # добавляем третье измерение канала к изображению
    image = np.expand_dims(image, axis=2)

    #забираем имя файла на основе папки
    label = image_file.split(os.path.sep)[-2]

    # добавляем изображение и его название в обучающую выборку
    data.append(image)
    labels.append(label)


# масштабируем необработанные интенсивности пикселей до диапазона [0, 1] (это улучшает обучение)
data = np.array(data)
labels = np.array(labels)

# раздиляем трейн и тест данные
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
#
# преобразование меток (букв)
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)
#X_train.info()



#Y_train.info()# сохранение разметки из знаков инкодированием
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)


model = Sequential()

# первый сверточный слой с максимальным объединением
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="PReLU"))#activation="PReLU"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# второй сверточный слой с максимальным объединением
model.add(Conv2D(50, (5, 5), padding="same", activation="PReLU"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# скрытый слой с 500 узлами
model.add(Flatten())
model.add(Dense(500, activation="PReLU"))
# выходной слой с 32 узлами (по одному для каждой возможной буквы/цифры, которую мы предсказываем)
model.add(Dense(32, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# обучение нейросети
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# записываем
model.save(MODEL_FILENAME)
