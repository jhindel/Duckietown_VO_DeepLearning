import cv2

img = cv2.imread("alex_2small_loops_images/frame000000.jpg")
area = (0, 160, 640, 480)
0
img = img[160:480, 0:640]

cv2.imshow('image',img)
cv2.waitKey(0)

