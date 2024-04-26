import cv2
import tkinter as tk
from PIL import Image, ImageTk

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

# Create Tkinter window
root = tk.Tk()
root.title("Icadio")

# Define canvas dimensions
canvas_width = 270
canvas_height = 600

# Create a canvas for displaying the video feed
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

def update_frame():
    success, img = cap.read()
    if success:
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

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

Font_tuple = ("Arial",10, "bold") 

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
button = tk.Button(root, image=button_img, relief=tk.FLAT,height=50, width=125)
button_window = canvas.create_window(30, 520, anchor=tk.NW, window=button)

button2 = tk.Button(root, image=button_img2, relief=tk.FLAT,height= 50, width=50)
button_window2 = canvas.create_window(190, 520, anchor=tk.NW, window=button2)

button2.configure(font = Font_tuple)
 
update_frame()
root.mainloop()

# Release resources
cv2.destroyAllWindows()
cap.release()
