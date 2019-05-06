import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
		roi_gray = gray[y:y+h, x:x+w]

		img_item = "1.png"
		cv2.imwrite(img_item, roi_gray)
		color = (255,0,0)
		cv2.rectangle(frame,(x, y),(x+w, y+h), color, 2)
		cv2.imshow('frame', frame)
		if cv2.waitKey(20) & OxFF == ord('q'):
			break


cap.release()
cv2.destroyAllWindows()
