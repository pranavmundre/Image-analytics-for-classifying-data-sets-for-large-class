
import os
import cv2
import numpy as np
from PIL import Image

detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# recongnizer = cv2.face.createLBPHFaceRecognizer();\
recognizer = cv2.face.LBPHFaceRecognizer_create()
path="dataSet";

def getImagesWithID(path):
	imagePaths=[os.path.join(path, f) for f in os.listdir(path)]
	# print (imagePaths)
	faces=[]
	IDs=[]
	for imagePath in imagePaths:
		faceImg=Image.open(imagePath).convert('L');
		faceNp=np.array(faceImg, 'uint8')
		ID=int(os.path.split(imagePath) [-1].split('.')[1])
		faces.append(faceNp)
		print(ID)
		IDs.append(ID)
		cv2.imshow("training", faceNp)
		cv2.waitKey(10)
	return IDs, faces

Ids, faces=getImagesWithID(path)
recognizer.train(faces, np.array(Ids))
recognizer.save("recognizer/trainningData.yml")
cv2.destroyAllWindow()

# getImageWithID(path)









# import cv2,os
# import numpy as np
# from PIL import Image

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# def getImagesAndLabels(path):
#     imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
#     faceSamples=[]
#     Ids=[]
#     for imagePath in imagePaths:
#         pilImage=Image.open(imagePath).convert('L')
#         imageNp=np.array(pilImage,'uint8')
#         Id=int(os.path.split(imagePath)[-1].split(".")[1])
#         faces=detector.detectMultiScale(imageNp)
#         for (x,y,w,h) in faces:
#             faceSamples.append(imageNp[y:y+h,x:x+w])
#             Ids.append(Id)
#     return faceSamples,Ids


# faces,Ids = getImagesAndLabels('dataSet')
# recognizer.train(faces, np.array(Ids))
# recognizer.save('recognizer/trainnerData.yml')




