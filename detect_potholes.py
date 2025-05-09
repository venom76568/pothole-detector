import cv2
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch the API key securely
api_key = os.getenv("api_key")

# Initialize the model
model = get_model(model_id="pothole-detection-yolov8/1", api_key=api_key)

# Function to detect potholes in a given frame
def detect_potholes(frame):
    # Perform inference on the frame using the model
    results = model.infer(frame)[0]

    # Convert results to Detections object
    detections = sv.Detections.from_inference(results)

    # Initialize annotators for bounding boxes and labels
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate the frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    return annotated_frame
