import keras
from keras.models import load_model
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        # Load Model
        self.model = load_model('Custom_model3.h5')
        self.labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Suprise']

    def __del__(self):
        self.video.release()        

    def get_frame(self,a):
            ret,i = self.video.read()
            gray=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
            face=self.cascade.detectMultiScale(gray,1.3,5)
            if a:

                for (x,y,w,h) in face:
                    cv2.rectangle(i,(x,y),(x+w,y+h),(0,0,255),3)

                if face !=():
                    frame = i[y:y + h, x:x + w]
                    img=cv2.imwrite('./live/test/live.jpg',frame)
                    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                        directory='./live/',target_size=(48,48),batch_size=1,
                        shuffle=False,color_mode='grayscale')
                    predict = self.model.predict(test_generator)
                    print(predict)
                    print(predict.argmax(axis=1))
                    l=self.labels[int(predict.argmax(axis=1))]
                    cv2.putText(img=i,text=l,org=(y,x+w),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,color= (0,0,255),thickness=3)
            ret, jpeg = cv2.imencode('.jpg', i)
            return jpeg.tobytes()