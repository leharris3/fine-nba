import pickle
import random
import logging
import ujson
import os
import lz4.frame
import msgpack
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from typing import Dict, List, Optional

logging.basicConfig(level=logging.WARN)


class PlayerPosition:

    def __init__(self, player_position: List):
        self.team_id: str = player_position[0]
        self.player_id: str = player_position[1]
        self.x: str = player_position[2]
        self.y: str = player_position[3]
        self.z: str = player_position[4]

    def to_dict(self) -> Dict:
        return {
            "team_id": self.team_id,
            "player_id": self.player_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


class Moment:

    def __init__(self, moment: List):
        self.period: str = moment[0]
        self.moment_id: str = moment[1]
        self.time_remaining_in_quarter: str = moment[2]
        self.shot_clock: str = moment[3]
        self.player_positions: List[PlayerPosition] = self.get_player_positions(
            moment[5]
        )

    def get_player_positions(
        self, player_positions: List[List]
    ) -> List[PlayerPosition]:
        player_positions_arr = []
        for pp in player_positions:
            player_positions_arr.append(PlayerPosition(pp))
        return player_positions_arr

    def to_dict(self) -> Dict:
        moment_dict = {
            "period": self.period,
            "moment_id": self.moment_id,
            "time_remaining": self.time_remaining_in_quarter,
            "shot_clock": self.shot_clock,
            "player_positions": [pp.to_dict() for pp in self.player_positions],
        }
        return moment_dict


class Event:

    def __init__(self, event: Dict):
        self.event_id: str = event["eventId"]
        self.visitor: str = event["visitor"]
        self.home: str = event["home"]
        self.moments: List[Moment] = self.get_moments(event["moments"])

    def get_moments(self, moments: List[List]) -> List[Moment]:
        moments_arr = []
        for moment in moments:
            moments_arr.append(Moment(moment))
        return moments_arr


class StatVUAnnotation:
    """
    Wrapper and helper functions for sportvu annotations for 15'-16' NBA games.
    Data Source: https://github.com/linouk23/NBA-Player-Movements
    """

    def __init__(self, fp: str):
        assert os.path.isfile(fp), f"Error: invalid path to statvu file: {fp}"
        assert fp.endswith(".json"), f"Error: invalid ext, expect .json for fp: {fp}"
        self.fp: str = fp
        with open(fp, "r") as f:
            self.data: Dict = ujson.load(f)

        self.gameid: str = self.data["gameid"]
        self.gamedata: str = self.data["gamedate"]
        self.events: List[Event] = self.get_events(self.data["events"])
        self.quarter_time_remaining_moment_map = (
            self.get_quarter_time_remaining_moment_map()
        )

    def get_events(self, events: List[Dict]) -> List[Event]:
        events_arr = []
        for event in events:
            events_arr.append(Event(event))
        return events_arr

    def get_quarter_time_remaining_moment_map(self) -> Dict[int, Dict[float, Moment]]:
        """
        {
            quarter: {
                time_remaining: Moment
            }
        }
        """

        quarter_time_remaining_map = {}
        for event in self.events:
            for moment in event.moments:
                period = moment.period
                time_remaining = moment.time_remaining_in_quarter
                if period not in quarter_time_remaining_map:
                    quarter_time_remaining_map[period] = {}
                quarter_time_remaining_map[period][time_remaining] = moment
        return quarter_time_remaining_map

    def find_closest_moment(self, val: float, period: int) -> Moment:
        """
        Return the closest `Moment` to a time remaining `val` in a game `period`.
        """

        tr_moments_map_subset: Dict[Moment] = self.quarter_time_remaining_moment_map[
            period
        ]

        keys: np.array = np.array(list(tr_moments_map_subset.keys())).astype(float)
        closest_idx: int = np.argmin(abs(keys - val))
        return tr_moments_map_subset[keys[closest_idx]]


class Bbox:
    
    def __init__(self, bbox: Dict) -> None:
        self.frame_number: int = bbox["frame_number"]
        self.player_id: int = bbox["player_id"]
        self.x: int = bbox["x"]
        self.y: int = bbox["y"]
        self.width: int = bbox["width"]
        self.height: int = bbox["height"]
        self.confidence: float = bbox["confidence"]
        self.keypoints: np.ndarray = bbox["keypoints"]
        self.bbox_ratio: np.ndarray = bbox["bbox_ratio"]
        
        
class VideoInfo:

    def __init__(self, video_info: Dict):
        self.caption: str = video_info["caption"]
        self.file_type: str = video_info["file_type"]
        self.video_fps: int = video_info["video_fps"]
        self.height: int = video_info["height"]
        self.width: int = video_info["width"]

class Frame:

    def __init__(self, frame: Dict):
        self.frame_id: int = frame["frame_id"]
        self.bbox: List[Bbox] = self.get_bboxes(frame["bbox"])
        self.tracklet = None
        
    def get_bboxes(self, bboxes: List[Dict]) -> List[Bbox]:
        bboxes_arr = []
        for bbox in bboxes:
            bboxes_arr.append(Bbox(bbox))
        return bboxes_arr

class ClipAnnotation:

    def __init__(self, data: Dict):
        self.video_id: int = data["video_id"]
        self.video_path: str = data["video_path"]
        self.frames: List[Frame] = self.get_frames(data["frames"])
        self.video_info: VideoInfo = VideoInfo(data["video_info"])
        
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

    DATASET_ROOT = "/mnt/mir/levlevi/nba-plus-statvu-dataset"
    CLIPS_DIR = "filtered-clips"
    ANNOTATIONS_DIR = "filtered-clip-annotations"
    THREE_D_POSES_DIR = "filtered-clip-3d-poses-hmr-2.0"
    STATVU_LOGS_DIR = "statvu-game-logs"

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
            
        self.clip_annotation = ClipAnnotation(self.annotation_data)
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

        self.statvu_aligned_fp = os.path.join(
            ClipAnnotation.DATASET_ROOT,
            "statvu-aligned",
            self.annotations_fp.split("/")[-3],
            self.annotations_fp.split("/")[-1],
        ).replace("_annotation", "")
        try:
            assert os.path.isfile(self.statvu_aligned_fp)
        except:
            logging.warning(
                f"statvu-aligned time-remaining results: {self.statvu_aligned_fp}, do not exist. Setting this attribute to None."
            )
            self.statvu_aligned_fp = None

        self.game_id: str = self.basename.split("_")[0]
        self.period: str = self.annotations_fp.split("/")[-2][-1]

        self.statvu_game_log_fp: Optional[str] = None
        statvu_log_file_paths = glob(
            os.path.join(
                ClipAnnotation.DATASET_ROOT, ClipAnnotation.STATVU_LOGS_DIR, "*", "*"
            )
        )
        for fp in statvu_log_file_paths:
            game_id = fp.split("/")[-2].split(".")[-1]
            if game_id == self.game_id:
                self.statvu_game_log_fp = fp
                break
        self.statvu_annotation: StatVUAnnotation = StatVUAnnotation(
            self.statvu_game_log_fp
        )

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
