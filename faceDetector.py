
import os
import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);

rec = cv2.face.LBPHFaceRecognizer_create();
# rec = cv2.createLBPHFaceRecognizer();
rec.read('recognizer/trainningData.yml');

id = 0
# font = cv2.InitFont(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL, 5, 1, 0, 4)
font = cv2.FONT_HERSHEY_SIMPLEX

while (True):
	ret, img = cam.read();
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x+w,y+h),(0,255,0), 2)
		id, conf = rec.predict(gray[y:y+h, x:x+w])
		if id == 1:
			id = "Pranav | Age=21"
		if id== 2:
			id = "Pranav | Age=21"
		if id== 3:
			id = "Aditya | Age=26"
		if id== 4:
			id = "Aditya | Age=26"
		if id == 5:
			id = "Pratik | Age = 21"
		if id == 6:
			id = "Unkwon"
		# if id == 1 or id == 2 or id == 3 or id == 4 :
		# 	pass
		# else:
		# 	id = "Unkwon"

		# cv2.PutText(img,str(id), (x, y+h), font, 255);
		# cv2.putText(img ,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)
		# cv2.PutText(cv2.fromarray(img),str(id), (x, y+h), font, 255);
		cv2.putText(img, id, (x, h+50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
	cv2.imshow("Image Analytics For Classing Datasets For Large Class", img);
	if cv2.waitKey(1) == ord('q') :
		break;
cam.release()
cv2.destroyAllWindow()
