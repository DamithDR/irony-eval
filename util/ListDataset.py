from datasets import Dataset


class ListDataset(Dataset):

    def __init__(self, data_list):
        self.original_list = data_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]