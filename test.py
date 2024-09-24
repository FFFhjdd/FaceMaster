import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import face_recognition
from PIL import Image, ImageDraw, ImageFont, ImageTk


def add_chinese_text(image, text, font_path='font/simsun.ttc', font_size=30, font_color=(0, 255, 0), position=(10, 10)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=font_color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def load_face_names(image_dir):
    face_images = os.listdir(image_dir)
    face_encodings = []
    face_names = []
    for face_image in face_images:
        name, _ = os.path.splitext(face_image)
        face_names.append(name)

        image_file = face_recognition.load_image_file(os.path.join(image_dir, face_image))
        face_encoding = face_recognition.face_encodings(image_file)[0]
        face_encodings.append(face_encoding)
    return face_encodings, face_names


class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别与表情检测")
        self.root.geometry("800x600")

        # 创建按钮的 Frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # 在 Frame 中放置按钮
        self.face_recognition_button = tk.Button(button_frame, text="人脸识别", command=self.start_face_recognition)
        self.face_recognition_button.pack(side=tk.LEFT, padx=10)

        self.emotion_detection_button = tk.Button(button_frame, text="表情检测", command=self.start_emotion_detection)
        self.emotion_detection_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(button_frame, text="终止", command=self.stop_camera)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.cap = None
        self.running = False

    def start_face_recognition(self):
        self.stop_camera()
        messagebox.showinfo("提示", "启动人脸识别")
        self.face_encodings, self.face_names = load_face_names('face_names')
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.process_frame(self.face_recognition_process)

    def start_emotion_detection(self):
        self.stop_camera()
        messagebox.showinfo("提示", "启动表情检测")
        self.model = load_model('models/emotion_detection_model.h5')
        self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.process_frame(self.emotion_detection_process)

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    def process_frame(self, process_function):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = process_function(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.video_label.after(10, lambda: self.process_frame(process_function))

    def face_recognition_process(self, image):
        face_locations = face_recognition.face_locations(image)
        face_encodings_in_image = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_in_image):
            matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
            name = '未知'
            if True in matches:
                first_match_index = matches.index(True)
                name = self.face_names[first_match_index]

            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            image = add_chinese_text(image, name, position=(left, top - 35))
        return image

    def emotion_detection_process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        emotion_labels = ['哀', '惊', '惧', '乐', '怒', '厌', '中']

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
            result = self.model.predict(reshaped_face)
            max_confidence = np.max(result)
            label = np.argmax(result)

            if max_confidence > 0.6:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                emotion = f'{emotion_labels[label]}：{max_confidence * 100:.2f}%'
                image = add_chinese_text(image, emotion, position=(x, y - 35))
            else:
                print(f'表情：{emotion_labels[label]}，置信度：{max_confidence * 100:.2f}%')

        return image


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
