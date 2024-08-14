import os
import random
from entities.clip_annotations import ClipAnnotation
from glob import glob

ann_dir = (
    "/mnt/mir/levlevi/nba-plus-statvu-dataset/filtered-clip-annotations-with-ratios-pkl"
)
all_fps = glob(ann_dir + "/*/*/*.pkl")
random.shuffle(all_fps)

dst_dir = "/mnt/mir/levlevi/nba-plus-statvu-dataset/__viz__"
for fp in all_fps[0:10]:
    ann = ClipAnnotation(fp)
    bn = ann.basename
    out_fp = os.path.join(dst_dir, f"{bn}.avi")
    ann.visualize_bounding_boxes(out_fp)
