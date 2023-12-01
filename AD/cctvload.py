import cv2
import os

# RTSP 주소
url = 'rtsp://210.99.70.120:1935/live/cctv028.stream'

# 프레임 저장 위치
out_path = "AD/save_img"

cap = cv2.VideoCapture(url)
count = 0

while(cap.isOpened()):
    ret, image = cap.read()
    cv2.imshow("video", image)
    cv2.waitKey(1)

    if(int(cap.get(1)) % 5 == 0):
        print('Saved frame number : ' + str(int(cap.get(1))))

        cv2.imwrite(os.path.join(out_path,f"{count}.png"),image)
        count += 1
        
cap.release()
