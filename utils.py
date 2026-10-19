import torch
import cv2
import numpy as np

class DeviceSelector:
    def select_device():
        """
        Determines and returns the best available device.
        """
        if torch.cuda.is_available():
            return torch.device(0)
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
class VideoCapture:
    def capture(input_path, output_path):
        
        cap = cv2.VideoCapture(input_path)
    
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        
        return cap, out
    

def initialize_camera_and_lines(configuration_file):

    file = open(configuration_file, "r")
    content = file.read()
    file.close()
    content = dict(eval(content))
    lines = content['lines']
    
    # Image parameters
    image_width = content['U']
    image_height = content['V']

    # Camera physical parameters
    focal_length = content['f']
    sensor_width = content['sw']
    sensor_height = content['sh']

    # Camera angles
    thyaw = content['thyaw'] # rotation on z angle
    throll = content['throll'] # rotation on y angle
    thpitch = content['thpitch'] # rotation on x angle

    # Position of the camera
    xc = content['xc']
    yc = content['yc']
    zc = content['zc']
    
    # From this point we swap the z coordinates with the y coordinates because 
    # openCV uses a different axis system with the y axis as the vertical axis 
    # and the z axis as the depth axis
    
    camera_point = np.array([xc,zc,yc])

    for i in lines:
        i.pop('id')

    x_line = []
    y_line = []
    for d in lines: # for every line
        i = 0
        for e in d: # for every element of the line
            if i % 2 == 0: # x coordinates
                x_line.append(d[e])
            else: # y coordinates
                y_line.append(d[e])
            i+=1

    points_line = np.array([[x, 0, y] for x, y in zip(x_line, y_line)], dtype=np.float32)

    # Pixel focal length
    fx = (image_width/sensor_width)*focal_length
    fy = (image_height/sensor_height)*focal_length

    # Optical center of th image
    cx = image_width/2
    cy = image_height/2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    # Distortion coefficients, we assume no distortion
    dist_coeffs = np.zeros(5)

    rotation_vector = np.array([thpitch, throll, thyaw])
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    translation_vector = -rotation_matrix @ camera_point
    line_points, _ = cv2.projectPoints(points_line, rotation_vector, translation_vector, camera_matrix, dist_coeffs, aspectRatio=fx/fy)
    line_points = line_points.reshape(len(x_line), 2)
    return image_height, image_width, line_points