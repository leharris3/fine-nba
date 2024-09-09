import os
import cv2
import torchvision
import pprint

from typing import Dict, List, Optional, Tuple

HEIGHT = 720
WIDTH = 1280
NUM_BBXS = 10


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

    def __init__(self, video_path: str) -> None:

        cap = cv2.VideoCapture(video_path)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = float(cap.get((cv2.CAP_PROP_FPS)))
        self.num_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        vframes, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
        self.num_frames_tv = vframes.shape[0]

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "num_frames_cv": self.num_frames_cv,
            "num_frames_tv": self.num_frames_tv,
        }


class ClipAnnotation:

    def __init__(self, video_path: str) -> None:

        self.video_path = video_path
        self.caption = self.get_caption_from_video_path(self.video_path)
        self.video_meta_data = VideoMetaData(self.video_path)

        # (T, 10, 4)
        self.bounding_boxes: List[BoundingBoxes] = self.get_bbxs()

    def get_bbxs(self) -> List[BoundingBoxes]:

        # HACK: loading bbx annotations from file paths
        tracklet_annotation_fp = self.video_path.replace(
            "filtered-clips-agressive-thresh",
            "filtered-clips-aggresive-thresh-tracklets",
        ).replace(".mp4", ".txt")

        assert os.path.isfile(tracklet_annotation_fp)

        tracklet_lines = []
        with open(tracklet_annotation_fp, "r") as f:
            tracklet_lines = f.readlines()

        # ensure that bbx entities are in a fixed, relative position in their list throught time
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
            "3-": "A basketball player missing a three-point shot",
            "assisting": "A basketball player assisting on a play",
            "screen": "A basketball player setting a screen",
            "rebound": "A basketball player grabbing a rebound",
            "turnover": "A basketball player committing a turnover",
            "1+": "A basketball player making a free throw",
            "1-": "A basketball player missing a free throw",
            "2+1": "A basketball player scoring and being fouled",
            "2-": "A basketball player missing a two-point shot",
            "2+": "A basketball player making a two-point shot",
            "foul": "A basketball player committing a foul",
            "pick'n'roll": "A basketball player executing a pick and roll",
            "post": "A basketball player posting up",
            "steal": "A basketball player stealing the ball",
            "technical foul": "A basketball player receiving a technical foul",
            "3+": "A basketball player making a three-point shot",
            "2f": "A basketball player committing their second foul",
            "3f": "A basketball player committing their third foul",
            "unsportmanlike foul": "A basketball player committing an unsportsmanlike foul",
            "3+1": "A basketball player making a three-pointer and being fouled",
            "second chance": "A basketball player getting a second chance opportunity",
            "2ft+": "A basketball player making two free throws",
            "2ft-": "A basketball player missing two free throws",
            "3ft+": "A basketball player making three free throws",
            "3ft-": "A basketball player missing three free throws",
        }
        key = video_path.split("/")[-1].split("_")[1]
        return caption_map[key] if key in caption_map else None

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "video_meta_data": self.video_meta_data.to_dict(),
            "bounding_boxes": [bbxs.to_dict() for bbxs in self.bounding_boxes],
        }


class ClipAnnotationWrapper:
    def __init__(self, video_fp: str):
        assert os.path.isfile((video_fp))
        self.video_fp = video_fp
        self.clip_annotation = ClipAnnotation(video_fp)


if __name__ == "__main__":
    fp = "/playpen-storage/levlevi/hq-basketball-dataset/filtered-clips-agressive-thresh/1038305_85_adelaide_36ers_89_new_zealand_breakers/period_1/329956951_3-_946.275_0.mp4"
    clip_wrapper = ClipAnnotationWrapper(fp)
    import json

    with open("t.json", "w") as f:
        json.dump(clip_wrapper.clip_annotation.to_dict(), f, indent=4)
