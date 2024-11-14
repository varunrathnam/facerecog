import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import torch
import torchvision.transforms as transforms
from PIL import Image
import time

# Load YOLOv5 Model
sys.path.append('./yolov5')
from models.experimental import attempt_load
from utils.general import non_max_suppression

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)  # Use webcam feed
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
            time.sleep(0.05)  # Small delay to reduce CPU usage
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class VandalismEvent:
    def __init__(self, person_id, bbox, bottle_bbox):
        self.person_id = person_id
        self.bbox = bbox
        self.bottle_bbox = bottle_bbox

class VandalismDetector:
    def __init__(self):
        self.model = attempt_load('yolov5s.pt')
        self.model.eval()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.last_detection_time = 0
        self.detection_interval = 1  # 1 second between detections

    def detect(self, frame):
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return []

        self.last_detection_time = current_time
        
        # Resize frame for detection
        resized_frame = cv2.resize(frame, (640, 640))
        img = self.transform(resized_frame).unsqueeze(0).to(self.device)

        # Inference
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)

        events = []
        if pred[0] is not None and len(pred[0]):
            people = []
            bottles = []
            for *xyxy, conf, cls in pred[0].cpu().numpy():
                if self.model.names[int(cls)] == 'person' and conf > 0.6:
                    people.append(xyxy)
                elif self.model.names[int(cls)] == 'bottle' and conf > 0.5:
                    bottles.append(xyxy)
            
            # Match persons with bottles
            for person_bbox in people:
                for bottle_bbox in bottles:
                    if self.is_holding(person_bbox, bottle_bbox):
                        events.append(VandalismEvent(None, person_bbox, bottle_bbox))
                        break
        return events

    def is_holding(self, person_bbox, bottle_bbox):
        # Check if bottle's center is within the person's bounding box
        bottle_center_x = (bottle_bbox[0] + bottle_bbox[2]) / 2
        bottle_center_y = (bottle_bbox[1] + bottle_bbox[3]) / 2
        return (person_bbox[0] < bottle_center_x < person_bbox[2] and
                person_bbox[1] < bottle_center_y < person_bbox[3])

class PersonTracker:
    def __init__(self):
        self.persons = {}
        self.next_id = 1

    def update(self, events):
        current_boxes = [event.bbox for event in events]
        matched_ids = []

        # Match current detections with known persons
        for event in events:
            best_match = None
            best_iou = 0
            for person_id, known_box in self.persons.items():
                iou = self.calculate_iou(event.bbox, known_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = person_id

            if best_match is not None and best_iou > 0.5:
                event.person_id = best_match
                self.persons[best_match] = event.bbox
                matched_ids.append(best_match)
            else:
                event.person_id = self.next_id
                self.persons[self.next_id] = event.bbox
                self.next_id += 1

        # Remove persons that are no longer tracked
        self.persons = {k: v for k, v in self.persons.items() if k in matched_ids}

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

class AlertSystem:
    def send_alert(self, event):
        print(f"ALERT: Person {event.person_id} detected holding a bottle. Potential vandalism!")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV Monitoring System")
        self.display_width = 640
        self.display_height = 480

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.vandalism_detector = VandalismDetector()
        self.person_tracker = PersonTracker()
        self.alert_system = AlertSystem()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        # Perform detection and alerting
        vandalism_events = self.vandalism_detector.detect(cv_img)
        self.person_tracker.update(vandalism_events)

        # Annotate the image
        for event in vandalism_events:
            self.alert_system.send_alert(event)

            # Draw bounding box for person
            x1, y1, x2, y2 = map(int, event.bbox)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, f"ID: {event.person_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw bounding box for bottle
            bx1, by1, bx2, by2 = map(int, event.bottle_bbox)
            cv2.rectangle(cv_img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        # Convert and display the image
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
