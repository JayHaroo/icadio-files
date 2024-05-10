import cv2
import tkinter as tk
from PIL import Image, ImageTk
from gtts import gTTS
import pygame
import io
from queue import Queue

terminate_key = ord('q')  # Press 'q' to terminate the program

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize pygame for audio playback
pygame.init()


# Create Tkinter window
root = tk.Tk()
root.title("Icadio")

# Define canvas dimensions
canvas_width = 270
canvas_height = 600

# Create a canvas for displaying the video feed
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Function to speak the detected objects
def speak():
    while not detected_objects.empty():
        object_name = detected_objects.get()
        tts = gTTS(text=object_name, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
            pygame.time.Clock().tick(10)  # Adjust the argument according to your preference

        pygame.mixer.music.wait()

# Function to update the frame and detect objects
def update_frame():
    success, img = cap.read()
    if success:
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) > 0:  # Check if classIds is not empty
            if isinstance(classIds, tuple):  # Check if classIds is a tuple
                classIds = classIds[0]  # Unpack the tuple

            for classId, confidence, box in zip(classIds, confs.flatten(), bbox):
                object_name = classNames[classId - 1].upper()
                detected_objects.put(object_name)

                # Draw bounding box and label on the image
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, object_name, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            print("No objects detected in the frame")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate scaling factor while maintaining aspect ratio
        height, width, _ = img.shape
        ratio = min(canvas_width / width, canvas_height / height)
        new_width = int(1600 * ratio)
        new_height = int(1900 * ratio)

        # Resize the image
        resized_img = cv2.resize(img, (new_width, new_height))

        img = Image.fromarray(resized_img)
        img = ImageTk.PhotoImage(image=img)

        canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor=tk.NW, image=img)
        canvas.image = img

    root.after(10, update_frame)

# Function to handle Listen button click
def listen_button_click():
    speak()

# Load the image
button_img = Image.open("Listen.png")
button_img = button_img.resize((160, 160))  
button_img = ImageTk.PhotoImage(button_img)

# Load the image
button_img2 = Image.open("Flash.png")
button_img2 = button_img2.resize((50, 50))  
button_img2 = ImageTk.PhotoImage(button_img2)

# Load the image
button_img3 = Image.open("Logo.png")
button_img3 = button_img3.resize((90, 90))  
button_img3 = ImageTk.PhotoImage(button_img3)

# Display the image on a label
label = tk.Label(root, image=button_img3)
label_window = canvas.create_window((canvas_width/2)-45,0, anchor=tk.NW, window=label)

# Add rounded rectangle button using Tkinter
button = tk.Button(root, image=button_img, relief=tk.FLAT, height=50, width=125, command=listen_button_click)
button_window = canvas.create_window(30, 520, anchor=tk.NW, window=button)

button2 = tk.Button(root, image=button_img2, relief=tk.FLAT, height=50, width=50)
button_window2 = canvas.create_window(190, 520, anchor=tk.NW, window=button2)

detected_objects = Queue()
update_frame()
root.mainloop()

# Release resources
cv2.destroyAllWindows()
cap.release()
