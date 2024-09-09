import cv2
import os
import json
import logging
import subprocess

from glob import glob
from scenedetect import detect
from statistics import mean
from typing import List, Tuple
from scenedetect import detect, HashDetector
from scenedetect.frame_timecode import FrameTimecode
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# TODO: where da' bugs at?!
MIN_SCENE_LEN = 2 * 30

# very aggressive threshold
THRESHOLD = 0.10

# TODO: hard-coded, could be causing issues
FPS = 30

# setup logging and log formatting
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_scene(video_fp: str) -> List[Tuple]:
    """
    Given a path to a clip, return a list of scenes objects.
    """

    assert os.path.isfile(video_fp), f"{video_fp} does not exist"
    logger.info(f"parsing scenes from: {video_fp}")

    def append_frame_length(interval):
        start_frame, end_frame = interval
        start_frame, end_frame = int(start_frame), int(end_frame)
        length_in_frames = end_frame - start_frame
        # return the original tuple with the frame length appended
        return (start_frame, end_frame, length_in_frames)

    detector = HashDetector(threshold=THRESHOLD)
    scene_list = detect(video_fp, detector)
    logger.debug(f"scenes: {scene_list}")

    # list of scenes and respective frame lengths
    scenes_with_frames = [append_frame_length(interval) for interval in scene_list]
    logger.debug(f"scenes_with_frames: {scenes_with_frames}")
    return scenes_with_frames


def filter_scenes(video_fp: str, scenes: List[Tuple]) -> List[Tuple]:
    """
    Return scenes that are:
        1. 2+ sec. in length
        2. contain an avg. of 3+ bbxs
    """

    logger.info(f"filtering scenes from: {video_fp}")
    assert os.path.isfile(video_fp), f"{video_fp} does not exist"

    # num frames to parse
    final_frame = int(cv2.VideoCapture(video_fp).get(cv2.CAP_PROP_FRAME_COUNT))
    assert final_frame > 0, f"{video_fp} has no frames"

    # filter scenes
    filtered_scenes = []
    for scene in scenes:
        
        # 1. longer than 2s?
        scene_start = int(scene[0])
        scene_end = int(scene[1])
        # length of a scene in seconds
        scene_len = scene_end - scene_start
        if scene_len < MIN_SCENE_LEN:
            logger.debug(f"scene too short: {scene_start} - {scene_end}")
            continue
        
        logger.debug(f"appending scene: {scene}")
        filtered_scenes.append(scene)

    logger.debug(f"filtered_scenes: {filtered_scenes}")
    return filtered_scenes


def create_new_clip(video_path: str, dst_path: str, scene) -> None:
    """
    Create a new (trimmed) clip from video_path to dst_path.
    """
    
    # skip creating new clip if it already exists
    if os.path.isfile(dst_path):
        return

    logger.info(f"creating new clip: {dst_path}")
    assert os.path.isfile(video_path), f"{video_path} does not exist"
    assert scene is not None, f"{scene} is  None"

    # save a new clip
    # videos tend to include a few frames w/ camera cut
    OFFSET = 15
    start_frame = int(scene[0])
    end_frame = int(scene[1]) - OFFSET
    logger.debug(f"start_frame: {start_frame}, end_frame: {end_frame}")
    
    # TODO: we assume the FPS is fixed
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_sec = start_frame / fps
    end_sec = end_frame / fps
    vid_len_sec = end_sec - start_sec
    logger.debug(f"start_sec: {start_sec}, end_sec: {end_sec}")

    subprocess.call(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_sec),
            "-i",
            video_path,
            "-c",
            "copy",
            "-t",
            str(vid_len_sec),
            "-b:v",
            "400M",
            "-crf",
            "0",
            dst_path,
        ]
    )


def _process_clip(fp: str) -> None:
    """
    Wrapper function for segmenting a single clip.
    """

    assert os.path.isfile(fp), f"{fp} does not exist"
    logger.info(f"processing clip: {fp}")
    
    # look for camera cuts
    scenes = parse_scene(fp)
    logger.debug(f"scenes: {scenes}")
    filtered_scenes = filter_scenes(fp, scenes)
    logger.debug(f"filtered_scenes: {filtered_scenes}")

    # throw out clips where we can't find a scene
    if len(filtered_scenes) == 0:
        return

    # TODO: hard-coded out dir
    video_dst_path = fp.replace("clips", "filtered-clips-agressive-thresh")
    video_dst_dir = os.path.dirname(video_dst_path)
    os.makedirs(video_dst_dir, exist_ok=True)

    assert os.path.isdir(video_dst_dir), f"{video_dst_dir} is an invalid dir"

    logger.debug(f"video_dst_dir: {video_dst_dir}")
    logger.debug(f"video_dst_path: {video_dst_path}")

    for scene_num, scene in enumerate(filtered_scenes):
        tmp_video_dst_path = video_dst_path.replace(".mp4", f"_{scene_num}.mp4")
        logger.debug(f"creating clip with scene num {scene_num}: {tmp_video_dst_path}")
        create_new_clip(fp, tmp_video_dst_path, scene)


def main():

    # all clip fps
    all_clip_file_paths = glob(
        "/playpen-storage/levlevi/hq-basketball-dataset/clips"
        + "/*/*/*.mp4"
    )
    logger.debug(f"found {len(all_clip_file_paths)} clips")
    with ThreadPoolExecutor(max_workers=64) as pool:
        pool.map(_process_clip, all_clip_file_paths)
        

if __name__ == "__main__":
    main()