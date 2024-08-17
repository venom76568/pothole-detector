import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

import detect_potholes as dp

# Initialize the Tkinter window
root = tk.Tk()
root.title("Pothole Detection App")
root.geometry("800x800")
root.configure(bg="#f3f3f3")  # Light gray color typical of Windows 11

# Add a title label
title_label = Label(
    root,
    text="Pothole Detection System",
    font=("Segoe UI", 32, "bold"),
    bg="#f3f3f3",
    fg="#333333",
)
title_label.place(relx=0.5, rely=0.08, anchor="center")

# Add a description label
description_label = Label(
    root,
    text="Enhance road safety with our real-time pothole detection system.",
    font=("Segoe UI", 14),
    bg="#f3f3f3",
    fg="#666666",
)
description_label.place(relx=0.5, rely=0.15, anchor="center")

# Label to display the video feed
video_label = Label(root, text="Get started by clicking the Start Detection button")
video_label.place(relx=0.5, rely=0.5, anchor="center")

cap = None


# Function to start the webcam feed
def start_video():
    global cap
    cap = cv2.VideoCapture(0)
    start_button.place_forget()  # Hide start button
    stop_button.place(relx=0.5, rely=0.9, anchor="center")  # Show stop button
    video_loop()


def video_loop():
    ret, frame = cap.read()
    if ret:
        # Detect potholes in the frame
        detected_frame = dp.detect_potholes(frame)

        # Convert the frame to ImageTk format
        img = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the image in the label
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, video_loop)  # Loop the video feed


def stop_video():
    global cap
    cap.release()
    cap = None
    video_label.config(image="")  # Clear the video feed
    stop_button.place_forget()  # Hide stop button
    start_button.place(relx=0.5, rely=0.9, anchor="center")  # Show start button


# Button to start the detection
start_button = Button(
    root,
    text="Start Detection",
    font=("Segoe UI", 16),
    command=start_video,
    bg="#0078d7",
    fg="#ffffff",
)
start_button.place(
    relx=0.5, rely=0.9, anchor="center"
)  # Initially center the start button

# Button to stop the detection (initially hidden)
stop_button = Button(
    root,
    text="Stop Detection",
    font=("Segoe UI", 16),
    command=stop_video,
    bg="#e81123",
    fg="#ffffff",
)
stop_button.place_forget()  # Initially hide the stop button

# Run the Tkinter event loop
root.mainloop()
