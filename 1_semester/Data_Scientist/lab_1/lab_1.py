import numpy
import cv2 as cv


img_name = "img/lab_1.jpg"
img_name2 = "img/lab_1_1.jpg"

# Загрузка оригинального изображения в цвете для всех заданий, кроме 2-го

img = cv.imread(img_name)
img2 = cv.imread(img_name2)


# Задание 1: Изменить размер изображения
small_img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
cv.imshow("Result", small_img)
print(small_img.shape)


# Задание 1: Изменение размера изображения
new_img = cv.resize(img, (427, 1080))
small_new_img = cv.resize(new_img, (new_img.shape[1] // 2, new_img.shape[0] // 2))
cv.imshow("Resize", small_new_img)


# Задание 2: Перевести изображение в черно-белое (только для этого задания)
img_grayscale = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
cv.imshow("Grayscale", img_grayscale)


# Задание 3: Размытие по Гауссу
blur = cv.GaussianBlur(img, (5, 5), 10)
small_blur = cv.resize(blur, (blur.shape[1] // 2, blur.shape[0] // 2))
cv.imshow("GaussianBlur", small_blur)


# Задание 3: Билатеральное размытие
blur2 = cv.bilateralFilter(img, 5, 40, 20)
small_blur2 = cv.resize(blur2, (blur2.shape[1] // 2, blur2.shape[0] // 2))
cv.imshow("bilateral blur", small_blur2)


# Задание 4: Наложение границ
border = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, None, value=0)
small_border = cv.resize(border, (border.shape[1] // 2, border.shape[0] // 2))
cv.imshow("Border", small_border)


img2 = cv.resize(img2, (img.shape[1], img.shape[0]))
small_img2 = cv.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2))


# Задание 5: Наложение одной картинки на другую
added_image = cv.addWeighted(img, 0.4, img2, 0.3, 50)
small_added_image = cv.resize(added_image, (added_image.shape[1] // 2, added_image.shape[0] // 2))
cv.imshow("Weighted", small_added_image)

# Задание 6: Поворот изображения на 90 градусов
rotated_image = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
small_rotated_image = cv.resize(rotated_image, (rotated_image.shape[1] // 2, rotated_image.shape[0] // 2))
cv.imshow("Rotated", small_rotated_image)


cv.waitKey(0)
