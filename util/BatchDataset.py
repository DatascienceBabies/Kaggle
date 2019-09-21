import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
import DatasetModifier
import Dataset

class BatchDataset:
    _csv_file_path = None
    _chunk_size = None
    _df_chunk = None
    _dataset_lines = None

    def _count_lines_in_file(self, filename):
        with open(filename) as f:
            return sum(1 for line in f)    

    def __init__(self, file_path, chunk_size):
        self._csv_file_path = file_path
        self._chunk_size = chunk_size
        self._df_chunk = pd.read_csv(file_path, chunksize=chunk_size)
        self._dataset_lines = self._count_lines_in_file(self._csv_file_path) - 1

    # Returns a list of chunk IDs
    #def chunk_ids(self):
    #    # Assume first line is parameter name line
    #    dataset_lines = _count_lines_in_file(self._csv_file_path) - 1

    #    return range(math.ceil(dataset_lines / self._chunk_size))

    def get_next_chunk(self):
        for chunk in self._df_chunk:
            dataset = Dataset.Dataset(True)
            dataset.load_dataset_from_pandas_dataset(chunk)
            return dataset

    def batch_amount(self):
        return int(np.ceil(self._dataset_lines / float(self._chunk_size)))