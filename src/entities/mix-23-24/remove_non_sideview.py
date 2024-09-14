import os
import json
import shutil

from glob import glob
from clip_annotation import ClipAnnotationWrapper
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

MIN_NUM_FRAMES = 200
MIN_AVG_BBXS = 8

ann_dir  = (
    "/playpen-storage/levlevi/hq-basketball-dataset/filtered-clips-annotations"
)
ann_fps = glob(os.path.join(ann_dir, "*", "*", "*.json"))

def process_fp(fp: str):
    
    # skip existing files
    dst_path = "filtered-clips-annotations", f"filtered-clips-annotations-{MIN_NUM_FRAMES}-frames"
    if os.path.isfile(dst_path):
        return
    
    with open(fp, 'r') as f:
        data = json.load(f)
    bbxs = data['bounding_boxes']
    bbx_per_frame = []
    for bb in bbxs:
        num_bbxs = 0
        for b in bb['bounding_box_instances']:
            if b is not None:
                num_bbxs += 1
        bbx_per_frame.append(num_bbxs)
    try:
        avg_bbxs = sum(bbx_per_frame) / len(bbx_per_frame)
        if avg_bbxs < MIN_AVG_BBXS:
            os.remove(fp)
            return
        elif data['video_meta_data']['num_frames'] < MIN_NUM_FRAMES:
            return
    except:
       os.remove(fp)
       return
       
    # create dst file par dir as needed
    os.makedirs(Path(dst_path).parent, exist_ok=True)
    # copy clip -> filtered data dir
    shutil.copy2(
        fp,
        dst_path
    )
        
with ProcessPoolExecutor(max_workers=64) as ex:
    futures = []
    for fp in tqdm(ann_fps):
        futures.append(ex.submit(process_fp, fp))
    for future in tqdm(as_completed(futures), total=len(ann_fps)):
        future.result()
