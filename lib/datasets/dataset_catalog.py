import sys

# sys.path.append('..')
# sys.path.append('../../')
# sys.path.append('../../../')
sys.path.append("../../")

# from lib.config.config import cfg


class DatasetCatalog(object):
    """訓練データセットの情報を格納した辞書データを持つクラス"""

    dataset_attrs = {
        "AngleDetectTrain_1": {
            "id": "angle",
            "data_root": "AngleDetection/TrainingData_1",
            "split": "train",
        },
        "AngleDetectVal_1": {
            "id": "angle",
            "data_root": "AngleDetection/TrainingData_1",
            "split": "val",
        },
        "AngleDetectTrain_2": {
            "id": "angle",
            "data_root": "AngleDetection/TrainingData_2",
            "split": "train",
        },
        "AngleDetectVal_2": {
            "id": "angle",
            "data_root": "AngleDetection/TrainingData_2",
            "split": "val",
        },
        "AngleDetectTest": {
            "id": "angle",
            "data_root": "AngleDetection",
            "split": "test",
        },
        "BrakeRotorsTrain": {
            "id": "Brake",
            "data_root": "BrakeRotors/casting_data/train",
            "split": "train",
        },
        "BrakeRotorsTest": {
            "id": "Brake",
            "data_root": "BrakeRotors/casting_data/test",
            "split": "test",
        },
        "COCO2017Train": {
            "id": "COCO",
            "data_root": "COCOdataset2017",
            "split": "train",
        },
        "COCO2017Val": {
            "id": "COCO",
            "data_root": "COCOdataset2017",
            "split": "val",
        },
        "COCO2017Test": {"id": "COCO", "data_root": "COCOdataset2017", "split": "test"},
        "LinemodTrain": {
            "id": "linemod",
            "data_root": "linemod/",
            "split": "train",
        },
        "LinemodTest": {
            "id": "linemod",
            "data_root": "linemod/",
            "split": "test",
        },
        "SampleTrain": {
            "id": "sample",
            # "data_root": "sample/train",
            "data_root": "sample",
            "split": "train",
        },
        "SampleTest": {
            "id": "sample",
            # "data_root": "sample/test",
            "data_root": "sample",
            "split": "test",
        },
        "Sample_2Train": {
            "id": "sample_2",
            "data_root": "sample_2/train",
        },
        "Sample_2Test": {
            "id": "sample_2",
            "data_root": "sample_2/test",
        },
    }

    @staticmethod
    def get(name: str) -> dict:
        """データセットごとに用意されている,
        `{データセット名: {'id': 'hoge', 'data_root': hoge, 'split': train or test}, }`
        の辞書データを読み出す関数

        Args:
            name (str): データセット名。事前にカタログに準備している名前のみ指定可能。

        Returns:
            dict: データセットの情報が保存された辞書
        """
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
