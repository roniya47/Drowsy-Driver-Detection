from flask import Flask, render_template, Response, redirect, url_for
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from keras.layers import Layer
from keras import backend as K

app = Flask(__name__, template_folder="template")


class MaxPooling2DLayer(Layer):
    def __init__(self, pool_size, strides, **kwargs):
        super(MaxPooling2DLayer, self).__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        self.strides = tuple(strides)

    def call(self, inputs):
        return K.pool2d(
            inputs,
            pool_size=self.pool_size,
            strides=self.strides,
            padding="valid",
            pool_mode="max",
        )


class DropoutLayer(Layer):
    def __init__(self, rate, **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return K.dropout(inputs, level=self.rate)


mixer.init()
sound = mixer.Sound("static/alarm.wav")

face = cv2.CascadeClassifier(
    "static/haar_cascade_files/haarcascade_frontalface_alt.xml"
)
leye = cv2.CascadeClassifier(
    "static/haar_cascade_files/haarcascade_lefteye_2splits.xml"
)
reye = cv2.CascadeClassifier(
    "static/haar_cascade_files/haarcascade_righteye_2splits.xml"
)

lbl = ["Close", "Open"]

model = load_model(
    "models/finalroni.h5",
    custom_objects={
        "MaxPooling2DLayer": MaxPooling2DLayer,
        "DropoutLayer": DropoutLayer,
    },
)
path = os.getcwd()
cap = None
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
start_time = time.time()
total_frames = 0
correct_predictions = 0


def generate_frames():  # continuosly captures frames from the camera
    global cap, font, count, score, thicc, rpred, lpred, start_time, total_frames, correct_predictions
    lbl = None
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]
            total_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )
            left_eye = leye.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            right_eye = reye.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            cv2.rectangle(
                frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED
            )

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

            for x, y, w, h in right_eye:
                r_eye = frame[y : y + h, x : x + w]
                count = count + 1
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(
                    r_eye, (92, 112)
                )  # Resize the image to match model input size
                r_eye = r_eye / 255
                r_eye = np.expand_dims(r_eye, axis=-1)  # Add channel dimension
                r_eye = np.expand_dims(r_eye, axis=0)  # Add batch dimension
                rpred = np.argmax(model.predict(r_eye), axis=-1)
                if rpred[0] == 1:
                    lbl = "Open"
                if rpred[0] == 0:
                    lbl = "Closed"
                break

            for x, y, w, h in left_eye:
                l_eye = frame[y : y + h, x : x + w]
                count = count + 1
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(
                    l_eye, (92, 112)
                )  # Resize the image to match model input size
                l_eye = l_eye / 255
                l_eye = np.expand_dims(l_eye, axis=-1)  # Add channel dimension
                l_eye = np.expand_dims(l_eye, axis=0)  # Add batch dimension
                lpred = np.argmax(model.predict(l_eye), axis=-1)
                if lpred[0] == 1:
                    lbl = "Open"
                if lpred[0] == 0:
                    lbl = "Closed"
                break

            if rpred[0] == 1 or lpred[0] == 1:
                score = score + 1  # Increases score when any eye is open
                cv2.putText(
                    frame,
                    "Closed",
                    (10, height - 20),
                    font,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if lbl == "Open":
                    correct_predictions += 1
            else:
                score = score - 1  # Decreases score when both eyes are closed
                cv2.putText(
                    frame,
                    "Open",
                    (10, height - 20),
                    font,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if lbl == "Closed":
                    correct_predictions += 1

            if score < 0:
                score = 0
            cv2.putText(
                frame,
                "Score:" + str(score),
                (100, height - 20),
                font,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            elapsed_time = time.time() - start_time
            accuracy = (correct_predictions / total_frames) * 100
            accuracy_text = "Accuracy: {:.2f}%".format(accuracy)
            cv2.putText(
                frame,
                accuracy_text,
                (10, height - 50),
                font,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if score > 15:
                cv2.imwrite(os.path.join(path, "image.jpg"), frame)
                try:
                    sound.play()
                except:
                    pass
                if thicc < 16:
                    thicc = thicc + 2
                else:
                    thicc = thicc - 2
                    if thicc < 2:
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/start_detection")
def start_detection():
    global cap
    cap = cv2.VideoCapture(0)  # Initialize the camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
    return render_template("aa.html")


@app.route("/stop_detection")
def stop_detection():
    global cap
    # cv2.destroyAllWindows()
    if cap is not None:
        cap.release()  # Release the camera object
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
