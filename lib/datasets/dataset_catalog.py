import sys

# sys.path.append('..')
# sys.path.append('../../')
# sys.path.append('../../../')
sys.path.append("../../")

# from lib.config.config import cfg


class DatasetCatalog(object):
    """訓練データセットの情報を格納した辞書データを持つクラス"""

    dataset_attrs = {
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
        "MnistTrain": {
            "id": "mnist",
            "data_root": "data",
            "split": "train",
        },
        "MnistTest": {
            "id": "mnist",
            "data_root": "data",
            "split": "test",
        },
        "SampleTrain": {
            "id": "sample",
            "data_root": "sample/train",
        },
        "SampleTest": {
            "id": "sample",
            "data_root": "sample/test",
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
