from boxmot import tracker_zoo
from pathlib import Path
from utils import *

class Tracker:
  def __init__(self):
    """
    Creates and returns an instance of the specified tracker type.
    
    Parameters:
    - tracker_type: The type of the tracker (e.g., 'strongsort', 'ocsort').
    - tracker_config: Path to the tracker configuration file.
    - reid_weights: Weights for ReID (re-identification).
    - device: Device to run the tracker on (e.g., 'cpu', 'cuda').
    - half: Boolean indicating whether to use half-precision.
    - per_class: Boolean for class-specific tracking (optional).
    - evolve_param_dict: A dictionary of parameters for evolving the tracker.
    
    Returns:
    - An instance of the selected tracker.
    """
    self.object_tracker = tracker_zoo.create_tracker(tracker_type= "hybridsort",
                                                     tracker_config= "configs/hybridsort.yaml",
                                                     reid_weights= Path('osnet_model\osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pt'), 
                                                     device= DeviceSelector.select_device(),
                                                     half= False,   
                                                     per_class= False,
                                                     evolve_param_dict= None)
    
  def tracks(self,detections, frame):
    """
    Perform tracking
    
    Params:
        - dets: a numpy array of detections in the format [[x1,y1,x2,y2,score, class_number]...]
        - img: the current frame
      
        Returns:
        - The similar array, where the last column is the object ID.
    """
    tracked_object = self.object_tracker.update(detections, frame)
    
    return tracked_object
      