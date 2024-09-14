import os
from pathlib import Path
import cv2
import torchvision
import pprint

from typing import Dict, List, Optional, Tuple

HEIGHT = 720
WIDTH = 1280
FPS = 30
NUM_BBXS = 10

VID_EXT = '.mp4'
TRACKLET_EXT = '.txt'

PLAYPEN_DIR = "/playpen-storage"
MNT_MIR_DIR = "/mnt/mir"
CLIPS_DIR = (
    "/playpen-storage/levlevi/hq-basketball-dataset/filtered-clips-aggressive-thresh"
)
TRACKLETS_DIR = "/playpen-storage/levlevi/hq-basketball-dataset/filtered-clips-aggressive-thresh-tracklets"

class BoundingBoxInstance:

    def __init__(self, row: str):
        (
            self.frame_idx,
            self.player_id,
            self.x1,
            self.y1,
            self.w,
            self.h,
            self.conf_score,
            _,
            _,
            _,
        ) = row.split(",")

        self.frame_idx = int(self.frame_idx)
        self.player_id = int(self.player_id)
        self.x1, self.y1, self.w, self.h = (
            int(float(self.x1)),
            int(float(self.y1)),
            int(float(self.w)),
            int(float(self.h)),
        )
        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h
        self.x1, self.x2 = self.x1 / WIDTH, self.x2 / WIDTH
        self.y1, self.y2 = self.y1 / HEIGHT, self.y2 / HEIGHT

    def to_dict(self):
        return {
            "frame_idx": self.frame_idx,
            "player_id": self.player_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "conf": self.conf_score,
        }


class BoundingBoxes:

    def __init__(
        self, frame_idx: int, rows: List[str], bbx_rel_idx, bbx_rel_idx_map
    ) -> None:

        self.frame_idx = frame_idx

        # (10, 4)
        self.bounding_box_instances: List[BoundingBoxInstance] = (
            self.get_bounding_box_instances(rows, bbx_rel_idx_map, bbx_rel_idx)
        )

    def get_bounding_box_instances(
        self, rows: List[str], bbx_rel_idx_map: Dict, bbx_rel_idx: int
    ) -> List[BoundingBoxInstance]:
        # pad w/ None
        # ensure that each unique entity has a fixed index over time
        bbx_instances = [None] * NUM_BBXS
        for row in rows:
            instance = BoundingBoxInstance(row)
            player_id = instance.player_id
            if player_id not in bbx_rel_idx_map:
                bbx_rel_idx_map[player_id] = bbx_rel_idx
                bbx_rel_idx += 1
            rel_idx = bbx_rel_idx_map[player_id]
            if rel_idx < NUM_BBXS:
                bbx_instances[rel_idx] = instance
        return bbx_instances

    def to_dict(self):
        return {
            "frame_index": self.frame_idx,
            "bounding_box_instances": [
                bbx.to_dict() if bbx else None for bbx in self.bounding_box_instances
            ],
        }


class VideoMetaData:

    def __init__(self, num_frames: Optional[int]) -> None:
        self.height = HEIGHT
        self.width = WIDTH
        self.fps = FPS
        self.num_frames = num_frames

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "num_frames": self.num_frames
        }


class ClipAnnotation:

    def __init__(self, video_path: str) -> None:

        self.video_path = video_path
        self.caption = self.get_caption_from_video_path(self.video_path)

        # (T, 10, 4)
        self.bounding_boxes = None
        try:
            self.bounding_boxes = self.get_bbxs()
        except:
            pass
        
        num_frames = -1 if self.bounding_boxes is None else len(self.bounding_boxes)
        self.video_meta_data = VideoMetaData(num_frames)

    def get_bbxs(self) -> List[BoundingBoxes]:
        # HACK: loading bbx annotations from file paths
        tracklet_annotation_fp = self.video_path.replace(
            str(Path(CLIPS_DIR).name),
           str(Path(TRACKLETS_DIR).name),
        ).replace(VID_EXT, TRACKLET_EXT)
        
        assert os.path.isfile(tracklet_annotation_fp)

        tracklet_lines = []
        with open(tracklet_annotation_fp, "r") as f:
            tracklet_lines = f.readlines()

        # ensure that bbx entities are in a fixed, 
        # relative position in their list throught time
        bbx_rel_idx, bbx_rel_idx_map = 0, {}

        bbxs = []
        current_frame_idx = 0
        current_frame_bbxs = []
        for line in tracklet_lines:
            line_arr = line.split(",")
            frame_idx, player_id, x1, y1, w, h, conf_score, _, _, _ = line_arr
            frame_idx = int(frame_idx)
            if frame_idx == current_frame_idx:
                current_frame_bbxs.append(line)
            else:
                bbxs.append(
                    BoundingBoxes(
                        current_frame_idx,
                        current_frame_bbxs,
                        bbx_rel_idx,
                        bbx_rel_idx_map,
                    )
                )
                current_frame_idx = frame_idx
                current_frame_bbxs = [line]
        return bbxs

    def get_caption_from_video_path(self, video_path: str) -> Optional[str]:
        caption_map = {
            "assisting": "A basketball player assisting on a play",
            "screen": "A basketball player setting a screen",
            "rebound": "A basketball player grabbing a rebound",
            "turnover": "A basketball player committing a turnover",
            "1+": "A basketball player making a free throw",
            "1-": "A basketball player missing a free throw",
            "2+1": "A basketball player scoring and being fouled",
            "2-": "A basketball player missing a two-point shot",
            "2+": "A basketball player making a two-point shot",
            "2f": "A basketball player committing their second foul",
            "2ft+": "A basketball player making two free throws",
            "2ft-": "A basketball player missing two free throws",
            "3-": "A basketball player missing a three-point shot",
            "3+": "A basketball player making a three-point shot",
            "3f": "A basketball player committing their third foul",
            "3ft+": "A basketball player making three free throws",
            "3ft-": "A basketball player missing three free throws",
            "foul": "A basketball player committing a foul",
            "pick'n'roll": "A basketball player executing a pick and roll",
            "post": "A basketball player posting up",
            "steal": "A basketball player stealing the ball",
            "technical foul": "A basketball player receiving a technical foul",
            "unsportmanlike foul": "A basketball player committing an unsportsmanlike foul",
            "3+1": "A basketball player making a three-pointer and being fouled",
            "second chance": "A basketball player getting a second chance opportunity",
        }
        key = video_path.split("/")[-1].split("_")[1]
        return caption_map[key] if key in caption_map else "A basketball player performs an action"

    def to_dict(self):
        bbxs = [bbxs.to_dict() for bbxs in self.bounding_boxes]
        return {
            "video_path": self.video_path,
            "video_meta_data": self.video_meta_data.to_dict(),
            "caption": self.caption,
            "bounding_boxes": bbxs,
        }

class ClipAnnotationWrapper:
    def __init__(self, video_fp: str):

        # open the video @video_fp and attempt to read a frame
        # set attributes to None if an error occurs or if video is blank
        try:
            assert os.path.isfile((video_fp))
            cap = cv2.VideoCapture(video_fp)
            ret, frame = cap.read()
            if not ret:
                raise Exception
        except:
            self.video_fp = None
            self.clip_annotation = None
            return

        self.video_fp = video_fp.replace(PLAYPEN_DIR, MNT_MIR_DIR)
        self.clip_annotation = ClipAnnotation(video_fp)
