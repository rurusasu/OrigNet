import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler


class ImageSizeBatchSampler(Sampler):
    def __init__(
        self,
        sampler: Sampler,
        batch_szie: int,
        drop_last: bool,
        min_size: int = 600,
        max_height: int = 800,
        max_width: int = 800,
        size_int: int = 8,
    ):
        """
        Args:
            sampler (torch.utils.data.sampler.Sampler):
            batch_size (int): バッチサイズ
            drop_last (bool): 最後の余ったデータを切り捨てるか
            min_size (int)
        """
        self.sampler = sampler
        self.batch_size = batch_szie
        self.drop_last = drop_last
        self.hmin = min_size
        self.hmax = max_height
        self.wmin = min_size
        self.wmax = max_width
        self.size_int = size_int
        self.hint = (self.hmax - self.hmin) // self.size_int + 1
        self.wint = (self.wmax - self.wmin) // self.size_int + 1

    def generate_height_width(self) -> float:
        """画像をリサイズするときの幅: w, 高さ: hをランダムに出力する関数"""
        hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class IterationBasedBatchSampler(BatchSampler):
    """
    BatchSamplerをラップして、指定された数のイタレーションがサンプリングされるまで、BatchSamplerからリサンプリングします。
    """

    def __init__(
        self, batch_sampler: BatchSampler, num_iterations: int, start_iter: int = 0
    ):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
