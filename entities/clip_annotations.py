import pickle
import random
import logging
import ujson
import os
import lz4.frame
import msgpack
import cv2

from tqdm import tqdm
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)


class ClipAnnotation:
    """
    Each clip in our dataset contains data scattered across many different files and data formats.
    This class is intended to simplify the process of parsing different annotations types for a single clip.
    """

    CLIPS_DIR = "filtered-clips"
    ANNOTATIONS_DIR = "filtered-clip-annotations"
    THREE_D_POSES_DIR = "filtered-clip-3d-poses-hmr-2.0"

    def __init__(self, annotation_fp: str):
        """
        Given a path to a primary-annotation file, derive the paths to all other annotations for a given clip.

        Params
        :annotation_fp: a path to a `.json` or `.pkl` file containing the primary annotations for each frame in a clip.
        """

        assert os.path.isfile(
            annotation_fp
        ), f"Error: {annotation_fp} is not a valid file"

        self.annotation_ext = ""
        self.annotation_data = None

        # subdir in level one of dataset
        self.subdir = annotation_fp.split("/")[-4]

        if annotation_fp.endswith(".json"):
            self.annotation_ext = ".json"
            with open(annotation_fp, "r") as f:
                self.annotation_data = ujson.load(f)
        elif annotation_fp.endswith(".pkl"):
            self.annotation_ext = ".pkl"
            with open(annotation_fp, "rb") as f:
                self.annotation_data = pickle.load(f)
        else:
            raise Exception(
                f"Invalid annotation file path extension, expected: ['.json', '.pkl']"
            )

        self.annotations_fp = annotation_fp
        self.basename = (
            os.path.basename(annotation_fp)
            .replace(self.annotation_ext, "")
            .replace("_annotation", "")
        )
        self.video_fp = (
            annotation_fp.replace(self.subdir, ClipAnnotation.CLIPS_DIR)
            .replace("_annotation", "")
            .replace(self.annotation_ext, ".mp4")
        )
        try:
            assert os.path.isfile(self.video_fp)
        except:
            logging.warn(
                f"Clip video file path: {self.video_fp}, does not exist. Setting this attribute to None."
            )
            self.video_fp = None

        self.three_d_poses_fp = annotation_fp.replace(
            self.subdir, ClipAnnotation.THREE_D_POSES_DIR
        ).replace(self.annotation_ext, "_bin.lz4")
        try:
            assert os.path.isfile(self.three_d_poses_fp)
        except:
            logging.warning(
                f"3D-pose file path: {self.three_d_poses_fp}, does not exist. Setting this attribute to None."
            )
            self.three_d_poses_fp = None

    def get_3d_pose_data(self):
        with lz4.frame.open(self.three_d_poses_fp, "rb") as compressed_file:
            # Step 2: Decompress the data
            compressed = compressed_file.read()
            compressed_data = lz4.frame.decompress(compressed)
        # Step 3: Deserialize using msgpack
        decompressed_data = msgpack.unpackb(compressed_data, raw=False)
        # Step 4: Handle any remaining tensor-like structures
        # Assuming that all tensor data was converted to lists, no further action is needed.
        return decompressed_data

    def get_frames(self):
        cap = cv2.VideoCapture(self.video_fp)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def visualize_bounding_boxes(self, dst_path: str):
        """
        Generate a visualization of player tracklets for an annotation to `dst_path`.
        """

        assert dst_path.endswith(
            ".avi"
        ), f"`dst_path` must have file ext '.avi', got: {dst_path}"

        # inefficient, but do I care? the answer is... no
        annotations = self.annotation_data
        frames = self.get_frames()

        player_id_colors_map = {}
        height, width, fps = 720, 1280, 30.0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

        for idx, frame in tqdm(
            enumerate(frames), desc="Generating Bounding Box Viz", total=len(frames)
        ):
            if idx >= len(annotations["frames"]):
                print(
                    # f"Idx: {idx} out of range of # frames for annotations at: {self.annotations_fp}.\nEnding viz early."
                )
                break
            frame_obj = annotations["frames"][idx]
            if "bbox" not in frame_obj:
                print(f"No `bbox` in frame object at idx: {idx}")
                writer.write(frame)  # write a blank frame
                continue
            bboxs = frame_obj["bbox"]
            for bbx in bboxs:
                if not "x" in bbx:
                    # print(f"Skipping invalid bbx: {bbx}")
                    continue
                player_id = bbx["player_id"]
                if not player_id in player_id_colors_map:
                    # assign each player a unique, dark color
                    player_id_colors_map[player_id] = (
                        random.randint(0, 255),
                        255,
                        random.randint(0, 255),
                    )
                color = player_id_colors_map[player_id]
                x, y, w, h = (
                    int(bbx["x"]),
                    int(bbx["y"]),
                    int(bbx["width"]),
                    int(bbx["height"]),
                )
                # print("x, y, w, h", x, y, w, h)

                # draw a bbx
                import numpy as np

                # print("frame shape: ", np.array(frame).shape)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)

                # add a label
                cv2.putText(
                    frame,
                    f"ID: {str(player_id)}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    4,
                )
            writer.write(frame)
        writer.release()

    @staticmethod
    def save_data_lz4(data: Dict, dst_fp: str):
        """
        Save a dict: `data` as an lz4 file with default compression to `dst_fp`.
        https://python-lz4.readthedocs.io/en/stable/quickstart.html#simple-usage
        """

        data_compressed = lz4.frame.compress(data)
        with lz4.frame.open(dst_fp, mode="wb") as f:
            f.write(data_compressed)
