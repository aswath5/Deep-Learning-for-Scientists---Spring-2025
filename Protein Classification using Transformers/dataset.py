import torch
from torch.utils.data import Dataset

class PAFDataset(Dataset):
    def __init__(self, dataframe, label_map=None):
        self.seqs = dataframe["sequence"].tolist()
        self.labels = None
        if "family_id" in dataframe.columns:
            self.labels = [label_map[i] for i in dataframe["family_id"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.seqs[idx]
        return self.seqs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        x, y = zip(*batch)
        return list(x), torch.tensor(y)
    return batch