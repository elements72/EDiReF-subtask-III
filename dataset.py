from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
from lightning import LightningDataModule
from pathlib import Path
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np



class UtteranceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

class MeldDataModule(LightningDataModule):
    def __init__(self, data_path='./data/', batch_size=16, num_workers=0):
        super().__init__()
        self.data_path = Path(data_path)

        self.dataset_path = self.data_path / 'MELD_efr.json'
        self.train_data_path = self.data_path / 'MELD_train_efr.json'
        self.val_data_path = self.data_path / 'MELD_val_efr.json'       
        self.test_data_path = self.data_path / 'MELD_test_efr.json'
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.emotion_encoder = LabelEncoder()
        # Datasets
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def pre_process(self, data):
        tmp = data.copy()
        tmp['emotions'] = data['emotions'].apply(lambda x: self.emotion_encoder.transform(x))
        tmp['triggers'] = data['triggers'].apply(lambda x: [0 if i == np.nan or i == None else i for i in x])
        tmp.drop(columns=['episode'], inplace=True)
        return tmp

    def prepare_data(self) -> None:
        generator = torch.Generator().manual_seed(42)
        # Check if val and test sets are available
        if not self.val_data_path.exists() or not self.test_data_path.exists() or not self.train_data_path.exists():
            print('Generating val and test sets...')
            data = pd.read_json(self.dataset_path)
            # Split of the train set
            train_data, val_data, test_data = random_split(data, [0.8, 0.1, 0.1], shuffle=True, generator=generator)
            # Save the data
            train_data.to_json(self.train_data_path, indent=2, orient="records", force_ascii=False)
            val_data.to_json(self.val_data_path, indent=2, orient="records", force_ascii=False)
            test_data.to_json(self.test_data_path, indent=2, orient="records", force_ascii=False)
        else:
            print('Val and test sets already exist.')

    def setup(self, stage=None) -> None:
        # Load the data
        print('Loading data...')
        self.train_data = pd.read_json(self.train_data_path)
        self.val_data = pd.read_json(self.val_data_path)
        self.test_data = pd.read_json(self.test_data_path)
        # Encode emotions
        emotions = self.train_data['emotions'].explode().unique()
        self.emotion_encoder.fit(emotions)
        print("Mapping:" + str(dict(zip(self.emotion_encoder.classes_, self.emotion_encoder.transform(self.emotion_encoder.classes_)))))
        # Preprocess data
        self.train_data = self.pre_process(self.train_data)
        self.val_data = self.pre_process(self.val_data)
        self.test_data = self.pre_process(self.test_data)
        # Create datasets
        self.train_dataset = UtteranceDataset(self.train_data)
        self.val_dataset = UtteranceDataset(self.val_data)
        self.test_dataset = UtteranceDataset(self.test_data)

    def collate(self, batch):
        padding_value_emotion = len(self.emotion_encoder.classes_)
        padding_value_trigger = 2
        speakers, emotions, utterances, triggers = zip(*batch)

        emotions = [torch.tensor(e, dtype=torch.long) for e in emotions]
        triggers = [torch.tensor(t, dtype=torch.long) for t in triggers]

        emotions = torch.nn.utils.rnn.pad_sequence(emotions, batch_first=True, padding_value=padding_value_emotion)
        triggers = torch.nn.utils.rnn.pad_sequence(triggers, batch_first=True, padding_value=padding_value_trigger)
        # Pad with a PAD sentence

        max_len_utterances = max([len(u) for u in utterances])
        for i, u in enumerate(utterances):
            for _ in range(max_len_utterances - len(u)):
                u.append('')

        return {
            'speakers': speakers,
            'emotions': emotions,
            'utterances': utterances,
            'triggers': triggers
        }        

    def train_dataloader(self, collate_fn=None):
        if collate_fn is None:
            collate_fn = self.collate
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers)
    
    def val_dataloader(self, collate_fn=None):
        if collate_fn is None:
            collate_fn = self.collate
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)
    
    def test_dataloader(self, collate_fn=None):
        if collate_fn is None:
            collate_fn = self.collate
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)
    
    
 

                