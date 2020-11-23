import joblib
import numpy as np
import cv2
import sys

d={ 1:"Paper",
    2:"Scissor",
    3:"Stone"
}

model = joblib.load("stone-paper-scissor.pkl")

img = cv2.imread(f"Test/{sys.argv[1]}.png")

out = img.copy()

img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img).reshape(1, -1)
# cv2.imshow("out",img)
# cv2.waitKey(0)
print(d[model.predict(img)[0]])