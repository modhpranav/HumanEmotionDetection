import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model(modelpath=None):
    if "h5" in modelpath:
        model = model_arch()
        model.load_weights(modelpath)
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
    else:
        model = pickle.load(open(modelpath, 'rb'))
    return model

def model_arch():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model

def infer(model):
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 2: "Happy", 3: "Sad"}

    # start the webcam feed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(haar_filter)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--haar",help="HAAR filter path (filename: haarcascade_frontalface_default.xml)")
    # ap.add_argument("--modeltype",help="Model Type (log/cnn)")
    ap.add_argument("--modelpath",help="CNN Model path")
    haar_filter = ap.parse_args().haar
    # modeltype = ap.parse_args().modeltype
    modelpath = ap.parse_args().modelpath
    if all([modelpath, haar_filter]):
        model = load_model(modelpath)
        infer(model)
    else:
        print("""
            Please pass files as mentioned below:
            
            In windows:
                python3 inference.py --haar "path\\to\\file\\named\\haarcascade_frontalface_default.xml" --modelpath "path\\to\\cnn\\model"
            
            In Linux/MacOS:
                python3 inference.py --haar "path/to/file/named/haarcascade_frontalface_default.xml" --modelpath "path/to/cnn/model"
            """)
