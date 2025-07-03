from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
import torch
import os
import numpy as np
class Processor:
    def __init__(self, model_path="yolo11n.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        YOLO_path = os.path.join(proj_path, "weights", "pretrained", model_path)
        self.model = YOLO(YOLO_path)
        self.device = device
        self.animal_ids = list(range(14, 24))  # COCO IDs for animals
        self.target_layers =[self.model.model.model[-2]]
    def run(self, frame):
        cam = EigenCAM(self.model, self.target_layers, task='od')
        rgb_frame = frame.copy()
        float_frame = np.float32(frame.copy())/255.0
        grayscale_cam = cam(rgb_frame)[0, :, :]
        results = self.model(frame, conf=0.3)[0] # Heuristic fine-tune
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id in self.animal_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "crop": crop
                })
        cam_image = show_cam_on_image(float_frame, grayscale_cam, use_rgb=True)
        return detections, cam_image
