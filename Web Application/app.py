from flask import Flask, render_template, Response, jsonify,request
from Camera import VideoCamera
import cv2
a = 0
app = Flask(__name__)

video_stream = VideoCamera()

@app.route('/',methods=['GET','POST'])
def index():
    global a
    if request.method == 'POST':
        if 'button_name' in request.form:
            if a == 0:
                a = 1
            elif a == 1:
                a = 0
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame(a)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
     return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port="5000")