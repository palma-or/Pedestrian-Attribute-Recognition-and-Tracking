from ultralytics import YOLO 
import numpy as np
  
class YoloDetector:
  """
  Creates and returns an instance of the specified detector type.
  
  Parameters:
    - model_path: Path to pretrained YOLO model
    - confidence: Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold are ignored
    - iou: Intersection Over Union (IoU) threshold for non-maximum suppression (NMS).
    - img_size: Defines the size of the image for inference
    - classes: Filters predictions based on a set of class IDs
    - half: Boolean indicating whether to use half-precision
    - device: Device to run the detector on (e.g., 'cpu', 'cuda').
    
    Returns:
    - An instance of the selected detector.
  """
  def __init__(self, model_path, confidence, iou, img_size, classes, half, device):
    self.model = YOLO(model_path, task = 'detect')
    self.confidence = confidence
    self.iou = iou
    self.classes = classes
    self.img_size = img_size
    self.half = half
    self.device = device

  def detect(self, image):
    """
    Performs detection
    
    Parameters:
    - image: Frame on which to perform detection
    
    Returns:
    - NumPy array containing: - The top-left corner of the bounding box
                              - The bottom-right corner of the bounding box
                              - confidence scores
                              - class IDs

    """
    results = self.model.predict(source= image, 
                                 conf=self.confidence,
                                 iou = self.iou,
                                 imgsz = self.img_size,
                                 classes = self.classes,
                                 half = self.half, 
                                 device = self.device ) 
    result = results[0]
    detections = self.make_detections(result)
    return detections  

  def make_detections(self, result):
    """
    Handles results from Yolo's predict method
    """
    result = result.cpu()
      
    boxes = result.boxes
    detections = []
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      class_number = int(box.cls[0].cpu().numpy())
      conf = box.conf[0].cpu().numpy()
      
      detections.append([x1, y1, x2, y2, conf, class_number]) 
      
    return np.array(detections)
