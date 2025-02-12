# modules

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import pytesseract

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions

import pyttsx3
import cv2

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# parameters
epochs = 10
learning_rate = 0.1
batch_size = 16
model = DenseNet121(weights='imagenet', include_top=False)

num_classes = 42  
output_layer = Dense(num_classes, activation='softmax')(model.output)

custom_model = Model(inputs=model.input, outputs=output_layer)

custom_model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


# app

app = tk.Tk()
app.geometry("300x300")
app.title("AccessiFood")

def take_snapshot():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('snapshot.jpg', frame)
    cap.release()

def preprocess_image(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expanded = np.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_expanded)
    return img_processed

def predict_image_class(processed_img):
    preds = custom_model.predict(processed_img)
    return preds

def detect_text(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def show_snapshot(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img)
    snapshot_label.config(image=img)
    snapshot_label.image = img

def image_prediction():
    take_snapshot()  
    path = 'snapshot.jpg'  
    input_img = preprocess_image(path)
    results = predict_image_class(input_img)
    text = detect_text(path)
    result_str = f"The product you have is: {results[0][1]} \n\nIt reads:\n{text}"
    result_label.config(text=result_str)


def convert_text_to_audio(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()



snapshot_button = tk.Button(app, text="Take Snapshot", command=image_prediction)
snapshot_button.pack(pady=10)

snapshot_label = tk.Label(app, bg='white')
snapshot_label.pack(pady=10)

result_label = tk.Label(app, text="", wraplength=300, justify="left")
result_label.pack(pady=10)

def quit_program(event=None):
    app.quit()

play_audio_button = tk.Button(app, text="Play Audio", command=lambda: convert_text_to_audio(result_label.cget("text")))
play_audio_button.pack(pady=10)

app.bind('q', quit_program)



train_data_dir = r"C:\Users\RAAGHAA\Downloads\Internship\project\AI\mosala.v1i.yolov8\train\images"
validation_data_dir = r"C:\Users\RAAGHAA\Downloads\Internship\project\AI\mosala.v1i.yolov8\valid\images"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')



# Training the model

history = custom_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)



app.mainloop()

