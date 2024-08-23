import random
import os
import cv2
from glob import glob
from tqdm import tqdm

CLASS_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    ":": 10,
    ".": 11
}

def main():
    data_dir = "data/yolo/synth-roi-dataset"
    train_dir = "data/yolo/train"
    val_dir = "data/yolo/val"
    
    image_file_paths = glob(os.path.join(data_dir, "*.png"))
    for img_fp in tqdm(image_file_paths, total=len(image_file_paths), desc="Doing some nasty stuff (: "):
        data_fp = img_fp.replace(".png", ".txt")
        assert os.path.isfile(img_fp)
        assert os.path.isfile(data_fp)
        
        img = cv2.imread(img_fp)
        img_height, img_width, _ = img.shape
        
        with open(data_fp, "r") as f:
            data = f.read()
        if len(data) == 0:
            os.remove(data_fp)
            os.remove(img_fp)
        data = data.split("\n")
        
        new_data_str = ""
        for idx, item in enumerate(data):
            if len(item) == 0: continue
            char, x1, y1, x2, y2 = item.split(" ")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x_center, y_center, width, height = int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2), x2 - x1, y2 - y1
            x_center_n, y_center_n, width_n, height_n = x_center / img_width, y_center / img_height, width / img_width, height / img_height 
            char_cls = CLASS_MAP[char]
            
            new_data_str += f"{char_cls} {x_center_n} {y_center_n} {width_n} {height_n}\n"
            
        dice_roll = random.random()
        subdir = ""
        if dice_roll < .20:
            subdir = val_dir
        else:
            subdir = train_dir
            
        img_path_basename = os.path.basename(img_fp)
        data_path_basename = os.path.basename(data_fp)
        
        img_out_fp = os.path.join(subdir, img_path_basename)
        data_out_fp = os.path.join(subdir, data_path_basename)
        cv2.imwrite(img_out_fp, img)
        with open(data_out_fp, 'w') as f:
            f.write(new_data_str)                                                        

if __name__ == "__main__":
    main()