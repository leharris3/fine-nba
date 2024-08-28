import os
from glob import glob

DATASET_ROOT = "/mnt/mir/levlevi/nba-plus-statvu-dataset"
FILTERED_CLIPS_ANN_DIR = "filtered-clip-annotations-with-video-info"
FILTERED_CLIPS_ANN_EXT = ".pkl"


class FilteredClipDataset:

    def __init__(self) -> None:
        self.filtered_clip_annotations_dir = os.path.join(
            DATASET_ROOT, FILTERED_CLIPS_ANN_DIR
        )
        assert os.path.isdir(
            self.filtered_clip_annotations_dir
        ), f"Error: {self.filtered_clip_annotations_dir} is not a valid dir!"
        self.filtered_clip_annotations_file_paths = glob(
            os.path.join(
                self.filtered_clip_annotations_dir,
                "*",
                "*",
                "*" + FILTERED_CLIPS_ANN_EXT,
            )
        )
        assert (
            len(self.filtered_clip_annotations_file_paths) > 0
        ), f"Error: could not find any files in {self.filtered_clip_annotations_dir}"

    def __init__(self, dir_path: str):
        self.filtered_clip_annotations_dir = dir_path
        assert os.path.isdir(
            self.filtered_clip_annotations_dir
        ), f"Error: {self.filtered_clip_annotations_dir} is not a valid dir!"
        self.filtered_clip_annotations_file_paths = glob(
            os.path.join(
                self.filtered_clip_annotations_dir,
                "*",
                "*",
                "*" + FILTERED_CLIPS_ANN_EXT,
            )
        )
        assert (
            len(self.filtered_clip_annotations_file_paths) > 0
        ), f"Error: could not find any files in {self.filtered_clip_annotations_dir}"
