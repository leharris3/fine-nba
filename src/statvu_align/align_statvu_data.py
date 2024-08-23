import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List
from entities.clip_annotations import ClipAnnotation
from entities.clip_dataset import FilteredClipDataset

# goal: map each frame idx to a float val representing the time remaining


def re_time_remaining(text: str):
    """
    Matches any string showing a valid time remaining of 20 minutes or less
    assumes brodcasts use MM:SS for times > 1 minute, and SS.S for times < 1 minute
    """

    if text is None:
        return None
    time_remaining_regex = r"(20:00)|(0[0-9]?:[0-9][0-9](\.[0-9])?)|([1-9]:[0-5][0-9])|(1[0-9]:[0-5][0-9](\.[0-9])?)|([0-9]\.[0-9])|([1-5][0-9]\.[0-9])"
    result = text.replace(" ", "")
    match = re.match(time_remaining_regex, result)
    if match is not None and match[0] == result:
        return result
    return None


def convert_time_to_float(time_remaining):
    """
    Coverts valid time-remaining str
    to float value representation.
    Return None if time-remaining is invalid.

    Ex: '1:30' -> 90.
    """

    if time_remaining is None:
        return None
    minutes, seconds = 0.0, 0.0
    if ":" in time_remaining:
        time_arr = time_remaining.split(":")
        minutes = float(time_arr[0])
        seconds = float(time_arr[1])
    elif "." in time_remaining:
        seconds = float(time_remaining)
    else:
        return None
    return (60.0 * minutes) + seconds


def get_time_remaining_from_result(result) -> Optional[str]:
    """
    Convert an ultralytics results obj to predicted time remaining.
    """

    names = result.names
    boxes = result.boxes.cpu().numpy()
    classes = boxes.cls.tolist()
    xyxy = boxes.xyxy.tolist()
    predicted_str = ""
    char_dict = {}
    for pred_class, box in zip(classes, xyxy):
        char = names[pred_class]
        x1 = box[0]
        char_dict[x1] = char

    # print(char_dict)
    for key in sorted(char_dict.keys()):
        predicted_str += char_dict[key]
    return re_time_remaining(predicted_str)


def interpolate_time_remaining(arr: List[float], fps: int = 30) -> List[float]:
    """
    Perform linear interpolatation of a "staircase" monotonically decreasing function.

    TL;DR: smoothly fill in the values between the whole number time-remaining vals scrapped off the game clock.
    """

    arr = arr.copy()
    grad_y = np.gradient(arr)
    inflection_points_y = np.argwhere(list(grad_y <= -0.3))[::2]
    interpolated_arr = np.array(arr)
    interpolated_arr = np.interp(
        np.arange(len(arr)),
        inflection_points_y[:, 0],
        interpolated_arr[inflection_points_y][:, 0] - 1,
    )
    return interpolated_arr


def convert_results_to_timeseries(results_fp: str) -> List[float]:

    assert results_fp.endswith(".pkl")
    with open(results_fp, "rb") as f:
        data = pickle.load(f)
    arr = []
    for _, result in enumerate(data):
        predicted_tr: str = convert_time_to_float(
            get_time_remaining_from_result(result)
        )
        arr.append(predicted_tr)

    # TODO: jank! handle no detected time remaining vals
    try:
        smoothed_results = interpolate_time_remaining(arr).tolist()
    except:
        return []
    return smoothed_results


def process_fp(fp: str):
    dst_dir = "__nba-plus-statvu-dataset__/filtered-clip-statvu-moments"
    ann = ClipAnnotation(fp)
    time_remaining = convert_results_to_timeseries(ann.statvu_aligned_fp)
    moments_frames_map = {}
    for frame_idx, tr in enumerate(time_remaining):
        moment = ann.statvu_annotation.find_closest_moment(tr, 4)
        moments_frames_map[frame_idx] = moment
    dst_path = os.path.join(dst_dir, '/'.join(ann.annotations_fp.split('/')[-3: ]))
    os.makedirs(Path(dst_path).parent.__str__(), exist_ok=True)
    with open(dst_path, 'wb') as f:
        pickle.dump(moments_frames_map, f)
    

def main():
    ann_dir = "/playpen-storage/levlevi/opr/fine-nba/src/statvu_align/__nba-plus-statvu-dataset__/filtered-clip-annotations-with-ratios-pkl"
    ann_fps = FilteredClipDataset(ann_dir).filtered_clip_annotations_file_paths
    with ProcessPoolExecutor(max_workers=64) as ex:
        futures = []
        for fp in ann_fps:
            futures.append(
                ex.submit(process_fp, fp)
            )
        for future in tqdm(as_completed(futures), total=len(ann_fps), desc="aligning statvu data"):
            try:
                future.result()
            except:
                print(f"oh fuck!")
        
if __name__ == "__main__":
    main()