from inference import get_model
import supervision as sv
import cv2

model = get_model(model_id="pothole-detection-yolov8/1", api_key="osxsMKLysLPLbK65nNI3")

image = cv2.imread("samples/samples (1).png")

results = model.infer(image)[0]


detections = sv.Detections.from_inference(results)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(image=annotated_image, size=(16, 16))
