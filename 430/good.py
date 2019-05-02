import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

img = cv2.imread("img2.jpg")
orig = img.copy()

# 定义HOG对象，采用默认参数，或者按照下面的格式自己设置
defaultHog = cv2.HOGDescriptor()
# 设置SVM分类器，用默认分类器
defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 这里对整张图片进行裁剪
# # detect people in the image
# (rects, weights) = defaultHog.detectMultiScale(img, winStride=(4, 4),padding=(8, 8), scale=1.05)
# for (x, y, w, h) in rects:
#     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
# rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
# pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
# for (xA, yA, xB, yB) in pick:
#     cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
# cv2.imshow("Before NMS", orig)
# cv2.imshow("After NMS", img)

# 只对ROI进行裁剪，img[height_begin:height_end,width_begin:width_end]
roi = img[0:200, 800:1600]
cv2.imshow("roi", roi)
cv2.imwrite("roi.jpg", roi)
(rects, weights) = defaultHog.detectMultiScale(roi, winStride=(4, 4), padding=(8, 8), scale=1.05)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(roi, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("roi", roi)
cv2.imwrite("roi_out.jpg", roi)
cv2.waitKey(0)