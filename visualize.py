import cv2

img_path = './data/person.jpg'
box = [0.787013, 0.571579, 0.305597, 0.481827]

img = cv2.imread(img_path)
img_h, img_w = img.shape[:2]

x, y, w, h = box
x_left = int((x - w / 2) * img_w)
y_top = int((y - h / 2) * img_h)
w = int(w * img_w)
h = int(h * img_h)

print(x_left, y_top, w, h)
cv2.rectangle(img, (x_left, y_top), (x_left + w, y_top + h), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey()
