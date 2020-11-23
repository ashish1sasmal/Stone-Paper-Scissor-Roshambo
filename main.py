# @Author: ASHISH SASMAL <ashish>
# @Date:   21-11-2020
# @Last modified by:   ashish
# @Last modified time: 23-11-2020

import cv2
from keras.models import load_model
import numpy as np


REV_CLASS_MAP = {
    0: "stone",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]

vid = cv2.VideoCapture(-1)

frame = None

while True:
    frame = vid.read()[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame,(30,30),(220,220),(30,230,10),2)
    frame = cv2.threshold(frame,180,255,cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("out",frame)
    k = cv2.waitKey(1)
    if k==32:
        # cv2.imwrite("test1.png",thres)
        break

img = frame
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = cv2.resize(img, (227, 227))

# cv2.imshow("out",img)
# cv2.waitKey(0)


model = load_model("stone-paper-scissors-model.h5")

# predict the move made
pred = model.predict(np.array([img]))
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
