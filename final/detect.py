# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import base64

import cv2
import requests
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def get_movement_direction(prev_coords, curr_coords):
    """
    Determines the direction of movement of an object based on its previous and current coordinates.

    Args:
      prev_coords: Tuple of x, y coordinates of the object in the previous frame.
      curr_coords: Tuple of x, y coordinates of the object in the current frame.

    Returns:
      A string representing the direction of movement of the object.
    """
    x_diff = curr_coords[0] - prev_coords[0]
    y_diff = curr_coords[1] - prev_coords[1]
    if y_diff > 0:
        return "Down"
    elif y_diff < 0:
        return "Up"
    else:
        return "None"


def run(
    model: str,
    camera_id: int,
    width: int,
    height: int,
    num_threads: int,
    enable_edgetpu: bool,
    url: str,
    expiry: dict
) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      model: Name of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      num_threads: The number of CPU threads to run the model.
      enable_edgetpu: True/False whether the model is a EdgeTPU model.
    """

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads
    )
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.50)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Variables to keep track of previous and current coordinates of detected objects
    prev_coords = {}
    coords = {}

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)
        
        for detection in detection_result.detections[:]:
            if 'person' in [category.category_name for category in detection.categories]:
                detection_result.detections.remove(detection)
                
        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = "FPS = {:.1f}".format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(
            image,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            text_color,
            font_thickness,
        )

        for i, detection in enumerate(detection_result.detections):
            # Get the bounding box coordinates
            bbox = detection.bounding_box
            
            x1, y1, x2, y2 = map(
                int,
                [bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height],
            )
            if coords:
                prev_coords = coords

            coords = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if prev_coords:
                direction = get_movement_direction(prev_coords, coords)
                retval, buffer = cv2.imencode('.jpg', image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                item = detection.categories[0].category_name
                exp = 0
                if item in expiry.keys():
                    exp = expiry[item]
                else:
                    exp = 864000
                if item in expiry.keys():
                    myobj = {'name': detection.categories[0].category_name, 'direction': direction, 'image': image_base64, 'expiry': exp}    
                    x = requests.post(url, data = myobj)
                #print(
                #    f"{detection.categories[0].category_name} detected at ({coords[0]}, {coords[1]}) moving {direction}"
                #)
            else:
                direction = "None"
                prev_coords = coords

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow("object_detector", image)

    cap.release()
    cv2.destroyAllWindows()


def main():
    url = 'http://192.168.10.47:5000/item'
    
    expiry = {'banana': 172800, 'apple': 604800, 'orange': 432000, 'bottle': 864000, 'donut': 86400}
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Path of the object detection model.",
        required=False,
        default="efficientdet_lite0.tflite",
    )
    parser.add_argument(
        "--cameraId", help="Id of camera.", required=False, type=int, default=0
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=640,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=480,
    )
    parser.add_argument(
        "--numThreads",
        help="Number of CPU threads to run the model.",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--enableEdgeTPU",
        help="Whether to run the model on EdgeTPU.",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    run(
        args.model,
        int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
        int(args.numThreads),
        bool(args.enableEdgeTPU),
        url,
        expiry
    )


if __name__ == "__main__":
    main()
