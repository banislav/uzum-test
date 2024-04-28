import pandas as pd
import torch

from torch.utils.data import Dataset


class ReturnsDataset(Dataset):
	def __init__(self, split):
		self.df = pd.read_csv(f'data/{split}_1.csv')
		self.shape = self.df.shape

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, index):
		row = self.df.iloc[index, :]
		x, target = row.drop(labels='target'), row['target']
		return torch.tensor(x.values), target