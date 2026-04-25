"""Machine-learning datasets, models, and training helpers for plume videos."""

__all__ = []

try:  # pragma: no cover - optional extra dependency
    from .build_model import Decoder, Encoder, ResNetBlock, VideoRegressionModel
    from .dataset_builder import (
        EqualRangeNormalizer,
        add_csv_columns_to_h5,
        add_h5_images_to_h5,
        make_dataset,
        merge_h5_and_csv,
    )
    from .hdf5_dataset import hdf5_dataset
    from .hdf5_video_dataset import hdf5_dataset_image, hdf5_dataset_video
    from .notebook_utils import split_train_valid, viz_dataloader
    from .trainer import ModelTrainer
except ModuleNotFoundError:  # pragma: no cover
    pass
else:  # pragma: no cover
    __all__.extend(
        [
            "Decoder",
            "Encoder",
            "EqualRangeNormalizer",
            "ModelTrainer",
            "ResNetBlock",
            "VideoRegressionModel",
            "add_csv_columns_to_h5",
            "add_h5_images_to_h5",
            "hdf5_dataset",
            "hdf5_dataset_image",
            "hdf5_dataset_video",
            "make_dataset",
            "merge_h5_and_csv",
            "split_train_valid",
            "viz_dataloader",
        ]
    )
