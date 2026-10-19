import argparse
import cv2
import time
import os
import numpy as np
import json
from torchvision import transforms
from PIL import Image
from skimage.draw import line
from screeninfo import get_monitors

from PAR.utils.utils import MaxQueue
from combinedModel import *
from yolo_detector import YoloDetector
from tracker import Tracker
from utils import *

video_global = ""
MODEL_PATH = "yolov8n.pt"                                 # model selection
VIDEO_PATH = "video/"                                     # video selection
FRAME_SAVE_PATH = ""                                      # path to store the frames processed
OUTPUT_PATH = "Detection_and_tracking/video_result.mp4"   # path to store the result (video)
script_dir = os.path.dirname(os.path.abspath(__file__))

class HandleVideo():
  """
  This class encapsulates the logic for processing a video using YOLOv11 object detection and DeepSort tracking.
  It tracks people, records their attributes, and monitors their movement in specified regions of interest.
      
  Attributes:
      - video (str):              Path to the video file.
      - configuration (str):      Path to the JSON configuration file specifying the coordinates of the lines.
      - results (str):            Path to store the results in JSON format.
      - device (torch.device):    Device for running the Vilt model on CPU or GPU.
      - people (dict):            Dictionary to store information about tracked individuals.
      - line1 (dict):             Dictionary for tracking line1 information, including counters and timers.
                                  Used to update the People's "line1_passages".
      - line2 (dict):             Dictionary for tracking line2 information, including counters and timers.
                                  Used to update the People's "line2_passages".
  """

  def __init__(self, video: str, configuration: str, results: str) -> None:
      """
      Initialize the HandleVideo class.

      Arguments:
          - video (str):          Path to the video file.
          - configuration (str):  Path to the JSON configuration file specifying lines coordinates.
          - results (str):        Path to store the results in JSON format.
      """
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.video = video
      self.configuration = configuration
      self.results = results
      self.people = {}
      global VIDEO_PATH
      VIDEO_PATH += video
      self.model = CombinedModel().to(self.device)
      self.model.load_state_dict(torch.load('model_RN34_SD.pth', map_location=self.device),strict=False)
      
      
  def check_passage(self, discrete_line_1, discrete_line_2, y_prev, lower_center, id, counting_lines1, counting_lines2):
    # Check if the person is crossing the line 1
          # Check if the person was behind the line
          check1 = False
          for dpoint in discrete_line_1:
            if y_prev[1] < dpoint[1] and y_prev[0] == dpoint[0]:
              check1 = True
              break
          # if was behind check if now is in front of the line
          if check1:
            for dpoint in discrete_line_1:
              if lower_center[1] >= dpoint[1] and lower_center[0] == dpoint[0]: # if subsequently the y is greater, given the same x then it has crossed the line
                self.people[id]["trajectory"].append(1)
                counting_lines1 += 1
                break
          
          check2 = False
          for dpoint in discrete_line_2:
            if y_prev[1] > dpoint[1] and y_prev[0] == dpoint[0]:
              check2 = True
              break
          # if was behind check if now is in front of the line
          if check2:
            for dpoint in discrete_line_2:
              if lower_center[1] <= dpoint[1] and lower_center[0] == dpoint[0]: # if subsequently the y is greater, given the same x then it has crossed the line
                self.people[id]["trajectory"].append(2)
                counting_lines2 += 1
                break

          
          return counting_lines1, counting_lines2

  def track(self):
    # Define the detector and the tracker that we want to use
    device = DeviceSelector.select_device()
    print(f"Using device: {device}")
    
    detector = YoloDetector(model_path=MODEL_PATH,
                            confidence=0.4, 
                            iou= 0.5, 
                            classes = 0,
                            img_size= 640, 
                            half = False, # set this to True to use half precision (speed up time, but less accuracy), it works only with GPU
                            device = device) 
    tracker = Tracker()

    cap, out = VideoCapture.capture(VIDEO_PATH, OUTPUT_PATH)

    if not cap.isOpened():
      print("Error: Unable to open video file.")
      exit()

    # To skip frames
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = round(fps_video / 5)  # Process only 5 frames per second
    frame_counter = 0 
      
    # DECOMMENT TO SAVE FRAMES
    # frame_idx = 0
    
    ret, frame = cap.read()
    # inizialize the camera structure and define the lines coordinates on the image
    image_height, image_width, line_coordinates = initialize_camera_and_lines(self.configuration)
    lines=[]
    
    for x in line_coordinates:
            u, v = int(x[0]), int(image_height-x[1])
            lines.append((u, v))

    center_lines = [(int((lines[0][0]+lines[1][0])/2), int((lines[0][1]+lines[1][1])/2)), 
                    (int((lines[2][0]+lines[3][0])/2), int((lines[2][1]+lines[3][1])/2)),
                    (int((lines[4][0]+lines[5][0])/2), int((lines[4][1]+lines[5][1])/2)),
                    (int((lines[6][0]+lines[7][0])/2), int((lines[6][1]+lines[7][1])/2))]
    
    discrete_line_1 = list(zip(*line(*lines[0], *lines[1])))
    discrete_line_2 = list(zip(*line(*lines[2], *lines[3])))
    
    previous_positions = {} # a dict for save the people movements
    counting_lines1 = 0
    counting_lines2 = 0

    start = time.time()
    # START THE VIDEO PROCESSING
    while True:
      ret, frame = cap.read()
      if not ret:
        break
        
      if frame_counter % frame_interval == 0:     # for skip frames
        
        start_time = time.perf_counter()
        detections = detector.detect(frame)
        tracked_objects = tracker.tracks(detections, frame)
        copiedFrame = frame.copy()
        # Draw the lines, the arrows an put the text on the frame
        cv2.line(frame, (lines[0]), (lines[1]), (172, 102, 73), 3) # line 1
        cv2.line(frame, (lines[2]), (lines[3]), (172, 102, 73), 3) # line 2
        
        cv2.arrowedLine(frame, center_lines[0],  # arrow 1
                        (center_lines[0][0], int(center_lines[0][1] + 50)),
                        (172, 102, 73), 3)
        cv2.arrowedLine(frame, center_lines[1], # arrow 2
                        (center_lines[1][0], int(center_lines[1][1] - 50)),
                        (172, 102, 73), 3)
        
        # Define two lines iterator for check the passage of people
        
        cv2.putText(img = frame, text = "1", org = (lines[0][0] + 10, lines[0][1] + 36), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0, color = (172, 102, 73), thickness = 3)
        cv2.putText(img = frame, text = "2", org = (lines[1][0] + 10, lines[1][1] + 36), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0, color = (172, 102, 73), thickness = 3)
        # Draw black background rectangle for information display
        cv2.rectangle(frame, (0, 0), (300, 100), (255,255,255), -1)
        
        id_list = [] # list of ids in the image
        
        for obj in tracked_objects:
          x1,x2,y1,y2 = int(obj[0]), int(obj[2]), int(obj[1]), int(obj[3])
          # draw the bounding box of the person and give the id
          cv2.rectangle(frame, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (0, 0, 255), 2)
          
          cv2.rectangle(frame, (int(obj[0]), int(obj[1])), (int(obj[0]) + 30, int(obj[1]) + 20), (255, 255, 255), -1)
          cv2.putText( frame, f"{str(int(obj[4]))}", (int(obj[0]) + 5, int(obj[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          # Bounding box coordinates
          lower_center = (int((x1 + x2) / 2), y2-30)
          cv2.circle(frame, lower_center, 2, (0, 255, 0), 2)
          # Extracted ID for the detected person.
          id_list.append(obj[4])
          id = obj[4]
          if id not in self.people:
              self.__populate(id)
              
          # Memorize the current position
          if obj[4] not in previous_positions:
            previous_positions.update({obj[4]:[]})
            
          # extract the previous position and compare with the actual position
          y_prev = previous_positions[obj[4]].pop() if len(previous_positions[obj[4]]) > 0 else lower_center
          previous_positions[obj[4]].append(lower_center)
          
          counting_lines1, counting_lines2 = self.check_passage(discrete_line_1, discrete_line_2, y_prev, lower_center, obj[4], counting_lines1, counting_lines2)
          # Extract the image portion within the bounding box for further processing.
          image_to_process = copiedFrame[y1:y2, x1:x2]

          # Update pedestrian attributes for the current detected ID using the extracted image portion.
          self.__update_par(image_to_process, id)
          # Retrieve and display pedestrian attributes on the GUI.
          pedestrian_attributes = self.__par_processing(id)
          
          cv2.rectangle(frame, (x1, y2), (x2, int(y2+100)), (255, 255, 255), -1)
          for i, t in enumerate(pedestrian_attributes):
            cv2.putText(img = frame,text = t, org = (x1, int(y2 + (i+1) * 30)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.7, color = (0, 0, 0), thickness = 1)
          
          cv2.putText(frame, f"{self.people[id].get('trajectory')}", (x1, int(y2+90)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        
        cv2.putText(frame, f"Total People: {len(tracked_objects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(frame, f"Passegges for line 1: {counting_lines1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(frame, f"Passegges for line 2: {counting_lines2}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Time to process detect and tarcking: {total_time}")
            
          # DECOMMENT TO SAVE FRAMES
          # frame_filename = os.path.join(FRAME_SAVE_PATH, f"frame_{frame_idx:04d}.jpg")
          # cv2.imwrite(frame_filename, frame)
          # frame_idx += 1
          
        out.write(frame)
        screen = get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        frame_height, frame_width = frame.shape[:2]
        scale_width = screen_width / frame_width
        scale_height = screen_height / frame_height
        scale = min(scale_width, scale_height)
        window_width = int(frame_width * scale)
        window_height = int(frame_height * scale)
        resized_frame = cv2.resize(frame, (window_width, window_height))
        cv2.imshow("Frame", resized_frame)
        
      # To skip frames
      frame_counter += 1

      key = cv2.waitKey(1) & 0xFF
      if key == ord("q") or key == 27:
        break

    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    
    end = time.time()
    print(f"Total time: {end - start}")
    self.final_people = {}
    self.final_people.update({'people':[]})
    for person_id, attributes in self.people.items():
      for item in attributes.keys():
        if item == 'id':
          attributes[item] = int(attributes[item])
        elif isinstance(attributes[item], MaxQueue):
          max_attribute = attributes[item].get_max()
          attributes[item] = max_attribute
      self.final_people['people'].append(attributes)
      
    with open(args.results, 'w') as json_file:
        json.dump(self.final_people, json_file, indent=2)
        
        
  def __par_processing(self, id: int) -> list:
        """
        Processes the PAR (Pedestrian Attribute Recognition) attributes for an identified individual, generating a list of strings describing the recognized attributes (gender, presence of bag or hat).

        Arguments:
            - id (int): The identifier of the individual whose attributes are being processed.

        Returns:
            - list: A list of strings that describe the individual's attributes.
        """
        genderText = "M" if self.people[id]['gender'].get_max() == "male" else "F" if self.people[id]['gender'].get_max() == "female" else "M"
        if self.people[id]['bag'].get_max() == False and self.people[id]['hat'].get_max() == False:
            bagText = "No Bag "
            hatText = "No Hat"
        elif self.people[id]['bag'].get_max() == True and self.people[id]['hat'].get_max() == False:
            bagText = "Bag"
            hatText = ""
        elif self.people[id]['bag'].get_max() == False and self.people[id]['hat'].get_max() == True:
            bagText = ""
            hatText = "Hat"
        elif self.people[id]['bag'].get_max() == True and self.people[id]['hat'].get_max() == True:
            bagText = "Bag "
            hatText = "Hat"
        return [f"Gender: {genderText}", f"{bagText}{hatText}"]   
        
  def __preprocess_image(self, image_np) -> torch.Tensor:
        """
        Preprocess an image for model inference. Converts the image from a NumPy array to a PyTorch tensor, applying resizing, normalization, and adding a batch dimension.

        Arguments:
            - image_np (numpy.ndarray): The image to preprocess, represented as a NumPy array.

        Returns:
            - torch.Tensor: The preprocessed image as a PyTorch tensor, ready for model input.
        """
        image = Image.fromarray(image_np.astype('uint8'), 'RGB')

        # Define the preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        
        # Apply the preprocessing pipeline
        image_tensor = preprocess(image)
        # Add a batch dimension (B x C x H x W)
        image_tensor = image_tensor.unsqueeze(0)        
        return image_tensor
    
  def __update_par(self, bbox: tuple, id : int) -> None:
        """
        Updates the PAR (Pedestrian Attribute Recognition) attributes for an individual based on the image corresponding to their bounding box. Performs inference using the MTANet model and updates the individual's attribute data in the `people` dictionary.

        Arguments:
            - bbox (tuple): The bounding box of the individual to process, containing coordinates (x, y, w, h).
            - id (int): The identifier of the individual.
        """
        self.model.eval()
        predictions = {}
        image_tensor = self.__preprocess_image(bbox)
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            image_tensor = image_tensor.to(self.device)
            gender_output, bag_output, hat_output = self.model(image_tensor)
            predictions['gender'] = (torch.sigmoid(gender_output) > 0.5).squeeze().tolist()
            predictions['bag'] = (torch.sigmoid(bag_output) > 0.5).squeeze().tolist()
            predictions['hat'] = (torch.sigmoid(hat_output) > 0.5).squeeze().tolist()
                    
        self.people[id]["gender"].add(MaxQueue.int_to_gender(predictions["gender"]), predictions["gender"])
        self.people[id]["bag"].add(MaxQueue.int_to_bag(predictions["bag"]), predictions["bag"])
        self.people[id]["hat"].add(MaxQueue.int_to_hat(predictions["hat"]), predictions["hat"])
        
  def __populate(self, id: int) -> None:
        """
        Initialize entries for a new person in the people dictionary and corresponding ROI timers.

        Arguments:
            - id (int): Person's identifier.
        """
        self.people[id] = {
                        "id":id,
                        "gender": MaxQueue(),
                        "hat": MaxQueue(),
                        "bag": MaxQueue(),
                        "trajectory": list(),
                    }


if __name__ == "__main__":
  # Pass the arguments from commandi line
  parser = argparse.ArgumentParser("Group 10 - AV Contest")
  parser.add_argument('-p', "--video", type=str, default='video.mp4', help="Choose a video")
  parser.add_argument("--configuration", type=str, default='configuration.txt', help = "Specify the coordiantes file")
  parser.add_argument("--results", type=str, default='results.txt', help = "Specify where to store results")
  args = parser.parse_args()
  # create the handler video and track the video
  handler = HandleVideo(args.video, args.configuration, args.results)
  results = handler.track()
