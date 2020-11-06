from tkinter import *
from mtcnn import MTCNN
import cv2
import numpy as np
from keras.models import load_model
from tkinter import filedialog
from functools import partial
from random import choice
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import tensorflow as tf
from PIL import Image
import pickle

r = Tk()
filename = []
detector = MTCNN()
arr = ['Allen', 'Anusha', 'Ashna', 'Austin', 'Geethu', 'Jerry', 'Manna',
       'Rahul', 'Shraddha', 'Sruthi']


def detection(face_pixel):
    required_size = (160, 160)
    image = Image.fromarray(face_pixel)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    loaded_model = pickle.load(open('model.pickle', 'rb'))
    detection_model = load_model('facenet_keras.h5')
    face_pixels = face_array
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = detection_model.predict(samples)
    in_encoder = Normalizer(norm='l2')
    prediction = in_encoder.transform(np.reshape(yhat[0], (1, -1)))
    x = loaded_model.predict(prediction)
    if arr[x[0]]==None:
        return -1
    return arr[x[0]]


def browseFiles():
    file = filedialog.askopenfilename(initialdir="/",
                                      title="Select a File",
                                      filetypes=(("video files",
                                                  "*.mp4*"),
                                                 ("all files",
                                                  "*.*")))
    filename.append(file)
    # Change label contents
    label_file_explorer.configure(text="File Opened: " + file)


def capture(path):
    model = load_model('model-007.model')
    # result = detector.detect_faces(img)
    labels_dict = {0: 'NO MASK', 1: 'MASK'}
    color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    cap = cv2.VideoCapture(path[-1])
    while True:
        # Capture frame-by-frame
        __, img = cap.read()
        cv2.imshow('Live Image', img)
        k = cv2.waitKey(25)
        # press space key to start recording
        if k % 256 == 32:
            frame = img
            break
        # Use MTCNN to detect faces
    ''' height, width = frame.shape[:2]
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row,end_col = int(height * .5), int(width)
    cropped_top = frame[start_row:end_row, start_col:end_col]'''
    result = detector.detect_faces(frame)
    name = ""
    if result!=[]:
        for person in result:
            x, y, w, h = person['box']
            # keypoints = person['keypoints']
            face = frame[y:y + h, x:x + w]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_img = gray
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          color_dict[label],
                          5)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            name = detection(face)
            if label==0:
                name=name+"(No Mask)"
            # display resulting frame
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if name == -1:
            label_name.configure(text="Person not identified!")
        else:
            label_name.configure(text="Name:"+name)
    else:
        label_name.configure(text="Error:No face recognized!")
    # When everything's done, release capture
    cap.release()
    cv2.destroyAllWindows()


r.title('Mask detection and identification')
w = Label(r, bg='#000fff000', text='Welcome to the detection! click to capture image!')
label_file_explorer = Label(r,
                            text="Select Video",
                            width=70, height=4,
                            )
label_name = Label(r,
                   text="",
                   width=70, height=4,
                   )
button_selectfile = Button(r,
                           text="Select File",
                           command=browseFiles)
button = Button(r, highlightbackground='#000000', text='Capture image', activeforeground='red', width=25,
                command=partial(capture, filename))

w.grid(column=1, row=1)
label_file_explorer.grid(column=1, row=2)
button_selectfile.grid(column=1, row=3)
button.grid(column=1, row=4)
label_name.grid(column=1, row=5)
r.mainloop()
