from ultralytics import YOLO


def main():
    dataset_fp = "/playpen-storage/levlevi/opr/fine-nba/src/statvu_align/data/roi-time-remaining-100-ft/roi_dataset.yaml"
    checkpoint_fp = "/playpen-storage/levlevi/opr/fine-nba/src/statvu_align/runs/detect/train/weights/best.pt"
    m = YOLO(checkpoint_fp)
    results = m.train(
        data=dataset_fp,
        epochs=100,
        imgsz=320,
    )


if __name__ == "__main__":
    main()
