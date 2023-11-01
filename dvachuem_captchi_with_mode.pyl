from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "sammple"


# Загрузите метки модели (чтобы мы могли преобразовать прогнозы модели в фактические буквы)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

#загружаем обученную модель
model = load_model(MODEL_FILENAME)

captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
#captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

for image_file in captcha_image_files:
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, 25, 255, L2gradient=False)

    #добавление отступов около изображения
    #image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    #преобразование в черно-белое
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #нахождение конткуров(потом можно заменить на детектор кенни)
    #contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # contours = contours[1] if imutils.is_cv3() else contours[0]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_image_regions = []
#cv2.imshow(image)
    # цикл для нахождения контуров каждой из 4 символов и экстракт букв

    for contour in contours:

        #получаем прямоугольник содержащий контур
        (x, y, w, h) = cv2.boundingRect(contour)


        #сравниваем ширину и высоту контура, чтобы обнаружить буквы, соединенные в один блок
        if w / h > 1.25:
# если это условие выполняется, то делим на 2 блока
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

#проверка на 4 символа, больше 4 работает не корректно
    '''''''''
    if len(letter_image_regions) != 4:
        continue
    '''''''''
#сортируем слево направо, как бтс
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    #создаем изображеие для выхода и список для хранения определенных символов
    output = cv2.merge([image] * 3)
    predictions = []


    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        print(x, y, w, h)
        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        #изменяем размер изображения буквы до 20x20 пикселей, чтобы соответствовало данным обучения
        #print(letter_image[:1])
        #cv2.imshow("img", letter_image)
        #cv2.waitKey()
        #letter_image = resize_to_fit(letter_image, 20, 20)

            #pixels.append(img)
        # добавлем пространства 4d
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # вызываем нейросеть
        prediction = model.predict(letter_image)
        # конвертим ne-hot-encoded prediction обратно
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # рисуем на капче что спрогнозили
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # выводим текст капчи
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    cv2.imshow("Output", output)
    cv2.waitKey()
