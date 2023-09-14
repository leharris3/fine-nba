from logging import config
import re
import cv2
import os
import pytesseract
from typing import List

from video import Video
from utilities.text_extraction.entities.roi import ROI
from utilities.files import File
from utilities.text_extraction.preprocessing import preprocess_image

PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CONFIG = "--psm 11 --oem 3"


class FrameTimestamp:

    def __init__(self, quarter=None, time_remaining=None) -> None:
        self.quarter: int = quarter
        self.time_remaining: float = time_remaining


class VideoTimestamps:

    def __init__(self, video=None) -> None:
        self.timestamps = {}
        self.video: Video = video

    def set_timestamp(self, frame_index: int, frame_timestamp: FrameTimestamp):

        timestamp = {'quarter': frame_timestamp.quarter,
                     'time_remaining': frame_timestamp.time_remaining}
        self.timestamps[frame_index] = timestamp

    def save_timestamps_to(self, path):

        assert not os.path.exists, f"Error: file found at path: {path}"
        File.save_json(self.timestamps, to=path)


def extract_timestamps_from_video(video: Video):
    """Return a dict {frame: [quarter, time_remaining]}"""

    return {}


def extract_timestamps_from_image(image, print_results=None) -> FrameTimestamp:
    """
    Return a dict {quarter: int | None, time_remaining: float | None} from an image.

        Note: Assumes that an image is cropped.
    """

    def convert_time_to_float(time: str) -> float:
        """Convert a formated time str to a float value representing seconds remaining in a basektball game."""

        result: float = 0.0
        if ':' in time:
            time_arr = time.split(':')
            minutes = time_arr[0]
            seconds = time_arr[1]
            result = (int(minutes) * 60) + int(seconds)
        elif '.' in time:
            time_arr = time.split('.')
            seconds = time_arr[0]
            milliseconds = time_arr[1]
            result = int(seconds) + float('.' + milliseconds)
        else:
            raise Exception(
                f"Error: Invalid format provided for seconds remaining.")
        return result

    # Optional: append path to tesseract to sys.
    # pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT

    time_remaining, quarter = None, None

    image = preprocess_image(image, save=False)
    extracted_text: str = pytesseract.image_to_string(
        image, config=TESSERACT_CONFIG)
    results: List[str] = re.split(r'[\s\n]+', extracted_text)

    # accepts strings like 11:30, 1:23, 10.2, 9.8
    time_remaining_regex = r'^(?:\d{1,2}:\d{2}|\d{1,2}\.\d)$'
    quarter_regex = r'^[1-4](st|nd|rd|th)$'

    # Check if the input string matches either format
    for result in results:
        result = result.lower()  # convert to lowercase
        if type(print_results) is bool and print_results:
            print(result)
        if re.match(time_remaining_regex, result):
            time_remaining = convert_time_to_float(result)
        elif re.match(quarter_regex, result):
            try:
                quarter = int(result[0])
            except:
                raise Exception("Error: invalid format provided for quarter.")

    return FrameTimestamp(quarter, time_remaining)


def is_valid_roi(frame, roi: ROI) -> bool:
    """Return True/False depending on if an ROI contains a valid game clock with legal values for quarter and time_remaining."""

    cropped_frame = frame[roi.y1: roi.y2, roi.x1: roi.x2]
    timestamp: FrameTimestamp = extract_timestamps_from_image(cropped_frame)
    print(timestamp.time_remaining, timestamp.quarter)
    if timestamp.quarter and timestamp.quarter:
        return True
    return False
