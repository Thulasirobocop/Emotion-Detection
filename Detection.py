import keras
from keras.models import load_model
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)

cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# Load Model
model = load_model('Custom_model3.h5')
vid = cv2.VideoCapture(0)
labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Suprise']

while(True):
	ret,i = vid.read()
	gray=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
	face=cascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in face:
		cv2.rectangle(i,(x,y),(x+w,y+h),(0,0,255),3)

	if face !=():
		frame = i[y:y + h, x:x + w]
		img=cv2.imwrite('./live/test/live.jpg',frame)
		test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
			directory='./live/',target_size=(48,48),batch_size=1,
			shuffle=False,color_mode='grayscale')
		predict = model.predict(test_generator)
		print(predict)
		print(predict.argmax(axis=1))
		l=labels[int(predict.argmax(axis=1))]
		print()
		cv2.putText(img=i,text=l,org=(y,x+w),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 1,color= (0,0,255),thickness=3)

	cv2.imshow('Camera',i)
	q=cv2.waitKey(1)
	if q==ord('q'):
		break

cv2.destroyAllWindows()