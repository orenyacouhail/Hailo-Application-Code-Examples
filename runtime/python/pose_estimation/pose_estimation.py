#!/usr/bin/env python3

import os
import sys
import queue
import threading
import argparse
from pathlib import Path
from PIL import Image
from loguru import logger

from pose_estimation_utils import *

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_input_images, validate_images, divide_list_to_batches

IMAGE_EXTENSIONS = ('.jpg', '.png', '.bmp', '.jpeg')


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Running a Hailo inference with actual images using Hailo API and OpenCV"
    )
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="yolov8s_pose_v2.hef"
    )
    parser.add_argument(
        "-i", "--input",
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-cn", "--class_num",
        help="The number of classes the model is trained on. Defaults to 1",
        default=1
    )

    args = parser.parse_args()
    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")
    return args


def create_output_directory() -> Path:
    """
    Create the output directory if it does not exist.

    Returns:
        Path: Path object for the output directory.
    """
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)
    return output_path


def enqueue_images(
    images: list[Image.Image],
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.

    Args:
        images (list[Image.Image]): list of PIL.Image.Image objects.
        batch_size (int): Number of images in one batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
    """
    # import ipdb; ipdb.set_trace()
    for batch in divide_list_to_batches(images, batch_size):
        processed_batch = []

        for image in batch:
            processed_image = preprocess(image, width, height)
            processed_batch.append(processed_image)

        input_queue.put(processed_batch)

    input_queue.put(None)


def process_output(
    output_queue: queue.Queue,
    output_path: Path,
    width: int,
    height: int,
    class_num: int,
    max_detections: int,
    score_threshold: int,
    nms_iou_thresh: int,
    regression_length: int,
    strides: list[int]
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
        class_num (int): Number of classes.
        max_detections (int): Maximal number of detections per class.
        score_threshold (float): Confidence threshold for filtering.
        nms_iou_thresh (float): IoU threshold for NMS.
        regression_length (int): Maximum regression value for bounding boxes.
        strides (list[int]): Stride values for each prediction scale.
    """
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        processed_image, raw_detections = result
        post_process_and_save_inference_results(
            processed_image, raw_detections, output_path, image_id,
            height, width, class_num,
            max_detections, score_threshold, nms_iou_thresh, regression_length, strides
        )
        image_id += 1

    output_queue.task_done()  # Indicate that processing is complete


class RaisingThread(threading.Thread):
  def run(self):
    self._exc = None
    try:
      super().run()
    except Exception as e:
      self._exc = e

  def join(self, timeout=None):
    super().join(timeout=timeout)
    if self._exc:
      raise self._exc


def infer(
    images: list[Image.Image],
    net_path: str,
    batch_size: int,
    class_num: int,
    output_path: Path,
    data_type_dict: dict,
    max_detections: int,
    score_threshold: float,
    nms_iou_thresh: float,
    regression_length: int,
    strides: list[int]
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        images (list[Image.Image]): list of images to process.
        net_path (str): Path to the HEF model file.
        batch_size (int): Number of images per batch.
        class_num (int): Number of classes.
        output_path (Path): Path to save the output images.
        data_type_dict (dict): Dictionary where the keys are layer names and values are a data type.
        max_detections (int): Maximal number of detections per class.
        score_threshold (float): Confidence threshold for filtering.
        nms_iou_thresh (float): IoU threshold for NMS.
        regression_length (int): Maximum regression value for bounding boxes.
        strides (list[int]): Stride values for each prediction scale.
    """
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, output_type=data_type_dict
    )
    height, width, _ = hailo_inference.get_input_shape()
    enqueue_thread = threading.Thread(
        target=enqueue_images,
        args=(images, batch_size, input_queue, width, height)
    )
    process_thread = RaisingThread(
        target=process_output,
        args=(
            output_queue, output_path, width, height, class_num,
            max_detections, score_threshold, nms_iou_thresh, regression_length, strides
        )
    )

    enqueue_thread.start()
    process_thread.start()

    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    try:
        process_thread.join()
    except KeyError as e: 
        logger.error(e)
        return
    else:        
        logger.info(
            f'Inference was successful! Results have been saved in {output_path}'
        )



def main() -> None:
    args = parse_args()
    images = load_input_images(args.input)

    try:
        validate_images(images, args.batch_size)
    except ValueError as e:
        logger.error(e)
        return

    output_path = create_output_directory()
    data_type_dict = data_type2dict(args.net, 'FLOAT32')
    infer(
        images, args.net, int(args.batch_size), int(args.class_num),
        output_path, data_type_dict,
        max_detections=300, score_threshold=0.001, nms_iou_thresh=0.7,
        regression_length=15, strides=[8, 16, 32]
    )

if __name__ == "__main__":
    main()
