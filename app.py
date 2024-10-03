import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import detect_potholes as dp

# Initialize the Tkinter window
root = tk.Tk()
root.title("Road Surface Irregularity Detector")
root.geometry("1200x800")
root.configure(bg="#f3f3f3")

# Initialize the global variable for video capture
cap = None

# Add a title label
title_label = Label(
    root,
    text="Road Surface Irregularity Detection",
    font=("Segoe UI", 32, "bold"),
    bg="#f3f3f3",
    fg="#333333",
)
title_label.pack(pady=(20, 10))

# Add a description label
description_label = Label(
    root,
    text="Enhance road safety with our real-time pothole detection system.",
    font=("Segoe UI", 14),
    bg="#f3f3f3",
    fg="#666666",
)
description_label.pack(pady=(0, 20))

# Create a frame for buttons
button_frame = tk.Frame(root, bg="#f3f3f3")
button_frame.pack(pady=(0, 20))


def reset_ui():
    # Show initial buttons and hide the close button
    image_button.pack(side="left", padx=10)
    video_button.pack(side="left", padx=10)
    live_button.pack(side="left", padx=10)
    close_button.pack_forget()
    empty_label.config(image="", text="Select an option to start detection")


def resize_image(image, target_height):
    # Resize image while maintaining aspect ratio
    (h, w) = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))


def detect_from_image():
    hide_initial_buttons()
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        detected_image = dp.detect_potholes(image)

        # Resize image to maintain aspect ratio
        resized_image = resize_image(detected_image, 600)

        # Convert the image to ImageTk format
        img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the image in the label
        empty_label.imgtk = imgtk
        empty_label.configure(image=imgtk)


def detect_from_video():
    global cap
    hide_initial_buttons()
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        video_loop(cap)


def detect_from_live_cam():
    global cap
    hide_initial_buttons()
    cap = cv2.VideoCapture(0)
    video_loop(cap)


def video_loop(capture):
    ret, frame = capture.read()
    if ret:
        detected_frame = dp.detect_potholes(frame)

        # Resize frame to maintain aspect ratio
        resized_frame = resize_image(detected_frame, 600)

        # Convert the frame to ImageTk format
        img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the image in the label
        empty_label.imgtk = imgtk
        empty_label.configure(image=imgtk)
        empty_label.after(10, lambda: video_loop(capture))
    else:
        capture.release()


def stop_detection():
    global cap
    if cap:
        cap.release()
        cap = None
    reset_ui()


def hide_initial_buttons():
    image_button.pack_forget()
    video_button.pack_forget()
    live_button.pack_forget()
    close_button.pack(side="left", padx=10)


# Buttons for different detection options
button_style = {
    "font": ("Segoe UI", 16),
    "bg": "#0078d7",
    "fg": "#ffffff",
    "width": 20,
    "height": 2,
    "bd": 0,  # No border
    "activebackground": "#005a9e",  # Darker shade of blue when pressed
    "activeforeground": "#ffffff",
}

image_button = Button(
    button_frame, text="Detect from Image", command=detect_from_image, **button_style
)
video_button = Button(
    button_frame, text="Detect from Video", command=detect_from_video, **button_style
)
live_button = Button(
    button_frame,
    text="Detect from Live Cam",
    command=detect_from_live_cam,
    **button_style
)
close_button = Button(
    button_frame,
    text="Close",
    font=("Segoe UI", 16),
    command=stop_detection,
    bg="#e81123",
    fg="#ffffff",
    width=20,
    height=2,
    bd=0,
    activebackground="#c50f1f",
    activeforeground="#ffffff",
)

# Create a frame for the output preview with padding
preview_frame = tk.Frame(
    root, width=600, height=600, bg="#ffffff", borderwidth=2, relief="solid"
)
preview_frame.pack_propagate(
    False
)  # Prevent the frame from resizing to fit its content
preview_frame.pack(pady=(10, 20))

# Label to display the video feed or processed image with padding
empty_label = Label(preview_frame, bg="#ffffff", padx=10, pady=10)
empty_label.place(relx=0.5, rely=0.5, anchor="center")

# Run the Tkinter event loop
reset_ui()  # Initialize the UI to its initial state
root.mainloop()
