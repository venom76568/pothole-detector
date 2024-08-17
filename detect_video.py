from inference import get_model
import supervision as sv
import cv2

# Initialize the model
model = get_model(model_id="pothole-detection-yolov8/1", api_key="osxsMKLysLPLbK65nNI3")

# Open the video file
video_path = "samples/pothole_vid.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
output_video_path = "output_video.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Optionally, display the frame (uncomment if needed)
    cv2.imshow("Pothole Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_video_path}")
