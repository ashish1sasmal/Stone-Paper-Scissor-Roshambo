# @Author: ASHISH SASMAL <ashish>
# @Date:   20-11-2020
# @Last modified by:   ashish
# @Last modified time: 21-11-2020

import cv2
import sys

i=0

flag = False

vid = cv2.VideoCapture(1)
while i!=int(sys.argv[2]):
    frame = vid.read()[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame,(30,30),(220,220),(30,230,10),2)
    thres = cv2.threshold(frame,180,255,cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("Out",thres)

    if flag:
        i+=1
        # cv2.imwrite(f"Images/{sys.argv[1]}/{sys.argv[1]}_{i}.png",thres[30:220,30:220])
        # print("pp")

    k = cv2.waitKey(1)
    if k==27:
        break
    elif k==32:
        flag = not flag

vid.release()
cv2.destroyAllWindows()

print(f"[{sys.argv[2]} {sys.argv[1]}s captured]")

img = cv2.imread("Images/paper/paper_1.png")
print(img.shape)
