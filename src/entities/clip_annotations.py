import pickle
import random
import logging
import ujson
import os
import lz4.frame
import msgpack
import cv2
import numpy as np

from pprint import pprint
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Optional
from .statvu import StatVUAnnotation

logging.basicConfig(level=logging.WARN)


class BoundingBox:

    def __init__(self, data: Dict, frame_number: int) -> None:

        self.frame_number: Optional[int] = (
            data["frame_number"] if "frame_number" in data else frame_number
        )
        self.player_id: int = data["player_id"]
        self.x: float = data["x"] if "x" in data else 0
        self.y: float = data["y"] if "y" in data else 0
        self.width: float = data["width"] if "width" in data else 0
        self.height: float = data["height"] if "height" in data else 0
        self.confidence: float = data["confidence"] if "confidence" in data else 0
        self.bbox_ratio: np.ndarray = data["bbox_ratio"]


class Frame:

    def __init__(self, data: Dict) -> None:
        self.frame_id: int = data["frame_id"]
        try:
            self.bbox: List[BoundingBox] = self.get_bounding_boxes(data["bbox"])
        except:
            pprint(data)
            assert False
        # TODO: we originally intended for tracklets to correspond to statvu 2d position `moment` data
        # we currently have this data kept in a seperate subdir
        # if we have values for some reason at data['tracklet'], they should be considered garbage and ignored
        self.tracklet = None

    def get_bounding_boxes(self, data: List[Dict]) -> List[BoundingBox]:
        bbox_arr = []
        for bbx in data:
            bbox_arr.append(BoundingBox(bbx, self.frame_id))
        return bbox_arr


class VideoInfo:

    def __init__(self, data: Dict) -> None:
        self.caption: str = data["caption"]
        self.file_type: str = data["file_type"]

        # TODO: these should really be floats
        self.video_fps: int = data["video_fps"]
        self.height: int = data["height"]
        self.width: int = data["width"]


class ClipAnnotation:

    def __init__(self, data: Dict, verbose: bool = False) -> None:

        if verbose:
            pprint(data)

        self.video_id: int = data["video_id"]
        self.video_path: str = data["video_path"]
        self.frames: Optional[List[Frame]] = (
            self.get_frames(data["frames"]) if "frames" in data else None
        )
        self.video_info: Optional[VideoInfo] = (
            VideoInfo(data["video_info"]) if "video_info" in data else None
        )

    def get_frames(self, frames: List[Dict]) -> List[Frame]:
        frames_arr = []
        for frame in frames:
            frames_arr.append(Frame(frame))
        return frames_arr


class ClipAnnotationWrapper:
    """
    Each clip in our dataset contains data scattered across many different files and data formats.
    This class is intended to simplify the process of parsing different annotations types for a single clip.
    """

    # TODO: dynamicaly set root to 'mnt' or 'playpen-storage depending on machine
    DATASET_ROOT = "/mnt/mir/levlevi/nba-plus-statvu-dataset"
    CLIPS_DIR = "filtered-clips"
    ANNOTATIONS_DIR = "filtered-clip-annotations"
    THREE_D_POSES_DIR = "filtered-clip-3d-poses-hmr-2.0"
    STATVU_LOGS_DIR = "statvu-game-logs"

    def __init__(
        self, annotation_fp: str, verbose: bool = False, load_statvu: bool = False
    ) -> None:
        """
        Given a path to a primary-annotation file, derive the paths to all other annotations for a given clip.

        Params
        :annotation_fp: a path to a `.json` or `.pkl` file containing the primary annotations for each frame in a clip.
        """

        assert os.path.isfile(
            annotation_fp
        ), f"Error: {annotation_fp} is not a valid file"
        assert annotation_fp.endswith(".json") or annotation_fp.endswith(
            ".pkl"
        ), f"Error: invalid ext, expect .json or .pkl for annotation_fp: {annotation_fp}"
        self.annotations_fp: str = annotation_fp

        # load annotation data dict
        self.annotation_data: Optional[Dict] = None
        if annotation_fp.endswith(".json"):
            with open(annotation_fp, "r") as f:
                self.annotation_data = ujson.load(f)
        elif annotation_fp.endswith(".pkl"):
            with open(annotation_fp, "rb") as f:
                self.annotation_data = pickle.load(f)
        else:
            raise Exception(
                f"Invalid annotation file path extension, expected: ['.json', '.pkl']"
            )

        # load `ClipAnnotation` data object
        self.clip_annotation = ClipAnnotation(self.annotation_data, verbose=verbose)

        # set the path to the corresponding video clip
        subdir: str = annotation_fp.split("/")[-4]
        self.video_fp: str = (
            annotation_fp.replace(subdir, ClipAnnotationWrapper.CLIPS_DIR)
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

        # set a few attributes derived from fp
        basename: str = (
            os.path.basename(annotation_fp)
            .replace(self.annotation_ext, "")
            .replace("_annotation", "")
        )
        self.game_id: str = basename.split("_")[0]
        self.period: str = self.annotations_fp.split("/")[-2][-1]

        # path to raw statvu game log
        self.statvu_game_log_fp: Optional[str] = None
        statvu_log_file_paths = glob(
            os.path.join(
                ClipAnnotationWrapper.DATASET_ROOT,
                ClipAnnotationWrapper.STATVU_LOGS_DIR,
                "*",
                "*",
            )
        )
        # find the corresponding statvu game log
        for fp in statvu_log_file_paths:
            game_id = fp.split("/")[-2].split(".")[-1]
            if game_id == self.game_id:
                self.statvu_game_log_fp = fp
                break

        # optionally load `StatVUAnnotation`
        self.statvu_annotation: Optional[StatVUAnnotation] = None
        if load_statvu:
            self.statvu_annotation: StatVUAnnotation = StatVUAnnotation(
                self.statvu_game_log_fp
            )

        # find path to statvu aligned moments dict
        self.statvu_aligned_fp: Optional[str] = self.annotations_fp.replace(
            "filtered-clip-annotations-40-bbx-ratios", "filtered-clip-statvu-moments"
        )
        if not os.path.isfile(self.statvu_aligned_fp):
            self.statvu_aligned_fp = None

        # find path to 3D-pose data
        self.three_d_poses_fp: Optional[str] = annotation_fp.replace(
            self.subdir, ClipAnnotationWrapper.THREE_D_POSES_DIR
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
        clip_ann_obj = self.clip_annotation
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
            bboxs = clip_ann_obj.frames[idx].bbox
            for bbx in bboxs:
                player_id = bbx.player_id
                if not player_id in player_id_colors_map:
                    # assign each player a unique, dark color
                    player_id_colors_map[player_id] = (
                        random.randint(0, 255),
                        255,
                        random.randint(0, 255),
                    )
                color = player_id_colors_map[player_id]
                x, y, w, h = (
                    int(bbx.x),
                    int(bbx.y),
                    int(bbx.width),
                    int(bbx.height),
                )

                # print("x, y, w, h", x, y, w, h)
                # draw a bbx

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

    def visualize_player_statvu_positions(self, dst_path: str):
        """
        Generate a visualization of 2d-player positions for an annotation to `dst_path`.
        """

        assert dst_path.endswith(
            ".avi"
        ), f"`dst_path` must have file ext '.avi', got: {dst_path}"

        clip_ann_obj = self.clip_annotation
        with open(self.statvu_aligned_fp, "rb") as f:
            statvu_aligned_data = pickle.load(f)

        frames = self.get_frames()

        img_fp = (
            "/playpen-storage/levlevi/opr/fine-nba/src/entities/2d-court-diagram.png"
        )
        court_diagram = cv2.imread(img_fp)
        court_diagram = cv2.resize(court_diagram, (800, 500))

        team_id_colors_map = {}
        height, width, fps = 720, 1280 * 2, 30.0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))
        xmax, ymax = 100, 50

        for idx, frame in tqdm(
            enumerate(frames),
            desc="Generating StatVU Player Position  Viz",
            total=len(frames),
        ):

            moment = statvu_aligned_data[idx]
            player_positions = moment["player_positions"]
            canvas = np.zeros((720, 1280 * 2, 3), dtype="uint8")
            # set left half of screen to the color white
            canvas[:, :1280, :] = 255
            # set right half of canvas to frame
            canvas[:, 1280:, :] = frame
            # draw the court on the left side of the frame
            canvas[100:600, 200:1000, :] = court_diagram
            # breakpoint()
            for pp in player_positions:
                team_id = pp["team_id"]
                player_id = pp["player_id"]
                x, y = pp["x"], pp["y"]
                if team_id not in team_id_colors_map:
                    # assign team a random color
                    if len(team_id_colors_map) == 0:
                        # green
                        team_id_colors_map[team_id] = (0, 255, 0)
                    elif len(team_id_colors_map) == 1:
                        # red
                        team_id_colors_map[team_id] = (0, 0, 255)
                    else:
                        # blue
                        team_id_colors_map[team_id] = (255, 0, 0)
                # find the normalized x and y positions of the player on the left side of the screen
                x_norm = x / xmax
                new_x = int(x_norm * 800) + 200
                y_norm = y / ymax
                new_y = int(y_norm * 500) + 100
                color = team_id_colors_map[team_id]
                # draw a circle
                cv2.circle(canvas, (new_x, new_y), 10, color, -1)
                # place a player id label above each circle
                cv2.putText(
                    canvas,
                    f"ID: {str(player_id)}",
                    (new_x, new_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    4,
                )
                # draw the boundaries of the court as a red rectangle
                cv2.rectangle(canvas, (200, 100), (1000, 600), (0, 0, 255), 5)
            writer.write(canvas)
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
