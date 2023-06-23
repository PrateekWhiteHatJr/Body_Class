import cv2

body_detect = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('walking.avi')

while True:
    ret, frame = cap.read()

    grayImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    bodies = body_detect.detectMultiScale(grayImg,1.2,3)
    
    for(x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
    cv2.imshow("Web cam", frame)
    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
