from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import json
from glob import glob
from clip_annotation import ClipAnnotationWrapper
from tqdm import tqdm
from pathlib import Path

vids_dir = (
    "/playpen-storage/levlevi/hq-basketball-dataset/filtered-clips-aggressive-thresh"
)
vids_fps = glob(os.path.join(vids_dir, "*", "*", "*.mp4"))


def process_fp(fp: str):
    dst_fp = fp.replace(
        "filtered-clips-aggressive-thresh",
        "filtered-clips-annotations",
    ).replace(".mp4", ".json")
    # if file already exists -> skip
    if os.path.isfile(dst_fp): 
        return
    ann = ClipAnnotationWrapper(fp).clip_annotation
    # ensure there exists a corresponding bbx annotation file
    if ann.bounding_boxes is None:
        return
    # don't include videos w/o a caption (e.g., "accurate pass")
    if ann.caption is None:
        return
    # to avoid overwriting og videos
    assert not dst_fp.endswith(".mp4")
    # optionally make the parent dir
    os.makedirs(str(Path(dst_fp).parent), exist_ok=True)
    with open(dst_fp, "w") as f:
        json.dump(ann.to_dict(), f, indent=4)


with ProcessPoolExecutor(max_workers=64) as ex:
    futures = []
    for fp in tqdm(vids_fps):
        futures.append(ex.submit(process_fp, fp))
    for future in tqdm(as_completed(futures), total=len(vids_fps)):
        future.result()