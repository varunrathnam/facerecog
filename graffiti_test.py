import cv2
import os
from ultralytics import YOLO

# Load both models with descriptive variable names
graffiti_model = YOLO('/Users/kaytee/Documents/BroCodes/VandalismDetection/runs/detect/detector 04-23-13-399/weights/best.pt')  # for graffiti
spitting_model = YOLO('/Users/kaytee/Documents/BroCodes/VandalismDetection/runs/detect/graffiti_detector3/weights/best.pt')  # for spitting

# Choose which model to use by assigning it to `model`
# You can change this variable to `spitting_model` if you want to detect spitting
model = graffiti_model  # or spitting_model

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create a directory to save screenshots
output_dir = '/Users/kaytee/Documents/BroCodes/VandalismDetection/screenshots'
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for naming the screenshots
screenshot_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB for YOLO model input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection using the selected model
    results = model.predict(rgb_frame, conf=0.5)

    # Initialize a flag to check if any bounding box is detected
    detected = False

    # Start with the original frame
    annotated_frame = frame.copy()

    # Process each result and check for detected objects
    for result in results:
        if result.boxes:  # Check if bounding boxes are present
            detected = True
            # Draw annotations on the frame
            annotated_frame = result.plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the annotated frame
    cv2.imshow('Graffiti Detection', annotated_frame)

    # If bounding boxes are detected, save the frame
    if detected:
        screenshot_filename = os.path.join(output_dir, f'screenshot_{screenshot_counter}.jpg')
        cv2.imwrite(screenshot_filename, annotated_frame)
        screenshot_counter += 1
        print(f"Saved screenshot: {screenshot_filename}")

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
