import cv2
import numpy as np
import time

from glob import glob
from ultralytics import YOLO
from typing import Optional, List
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils.logging import set_verbosity_error

from utils.draw_bbx import draw_bbx
from entities.clip_annotations import ClipAnnotation
from entities.clip_dataset import FilteredClipDataset

# silence pesky transformer lib warnings
set_verbosity_error()

ROI_MODEL_FP = "models/roi.pt"
YOLO_TIME_REMAINING_KEY = 1

def speed_test(model: VisionEncoderDecoderModel, processor: TrOCRProcessor):
    
    num = 20
    start = time.time()
    for _ in range(num):

        # assuming HWC
        img = np.random.random((100, 200, 3))
        pixel_values = processor(img, return_tensors="pt").pixel_values

        # recognize text (generatively)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            max_new_tokens=10,
            clean_up_tokenization_spaces=True,
        )[0]

    print(f"Processed {num} frames in {time.time() - start}")

    # # use batch processing
    # start = time.time()
    # imgs = [np.random.random((100, 200, 3)) for _ in range(num)]
    # pixel_values = processor(imgs, return_tensors="pt").pixel_values
    # # recognize text (generatively)
    # generated_ids = model.generate(pixel_values)
    # _ = processor.batch_decode(
    #     generated_ids,
    #     skip_special_tokens=True,
    #     max_new_tokens=10,
    #     clean_up_tokenization_spaces=True,
    # )
    # print(f"Batch processed {num} frames in {time.time() - start}")

def main():
    # 1. load an example
    # 2. load the model and infer
    # 3. viz results

    # load pre-trained tr-ocr models
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-small-handwritten",
        max_length=10,
        clean_up_tokenization_spaces=True,
    )
    text_rec_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    speed_test(text_rec_model, processor)

if __name__ == "__main__":
    main()