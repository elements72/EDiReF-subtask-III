from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import LabelEncoder  

from sklearn.model_selection import train_test_split

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
        # Check if val and test sets are available
        if not self.val_data_path.exists() or not self.test_data_path.exists() or not self.train_data_path.exists():
            # print('Generating val and test sets...')
            data = pd.read_json(self.dataset_path)

            # Split of the train set
            train_data, val_test = train_test_split(data, test_size=0.2, shuffle=True)
            # Split of the val and test sets
            val_data, test_data = train_test_split(val_test, test_size=0.5, shuffle=True)
            # Convert to pandas dataframe
            train_data = pd.DataFrame(train_data, columns=data.columns)
            val_data = pd.DataFrame(val_data, columns=data.columns)
            test_data = pd.DataFrame(test_data, columns=data.columns)
            # Save the data
            train_data.to_json(self.train_data_path, indent=2, orient="records", force_ascii=False)
            val_data.to_json(self.val_data_path, indent=2, orient="records", force_ascii=False)
            test_data.to_json(self.test_data_path, indent=2, orient="records", force_ascii=False)
        else:
            # print('Val and test sets already exist.')
            pass

    def setup(self, stage=None) -> None:
        # Load the data
        # print('Loading data...')
        self.train_data = pd.read_json(self.train_data_path)
        self.val_data = pd.read_json(self.val_data_path)
        self.test_data = pd.read_json(self.test_data_path)
        # Encode emotions
        emotions = self.train_data['emotions'].explode().unique()
        self.emotion_encoder.fit(emotions)
        # print("Mapping:" + str(dict(zip(self.emotion_encoder.classes_, self.emotion_encoder.transform(self.emotion_encoder.classes_)))))
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
        batch_speakers, batch_emotions, batch_utterances, batch_triggers = zip(*batch)

        batch_emotions = [torch.tensor(e, dtype=torch.long) for e in batch_emotions]
        batch_triggers = [torch.tensor(t, dtype=torch.long) for t in batch_triggers]

        batch_emotions = torch.nn.utils.rnn.pad_sequence(batch_emotions, batch_first=True,
                                                         padding_value=padding_value_emotion)
        batch_triggers = torch.nn.utils.rnn.pad_sequence(batch_triggers, batch_first=True,
                                                         padding_value=padding_value_trigger)
        # Pad with a PAD sentence

        max_len_utterances = max([len(u) for u in batch_utterances])

        new_batch_utterances = []

        for i, (utterances, speakers) in enumerate(zip(batch_utterances, batch_speakers)):
            # copy the utterances, this is necessary because list are mutable
            utterances_copy = utterances.copy()

            for i in range(len(utterances_copy)):
                utterances_copy[i] = speakers[i] + ': ' + utterances_copy[i]

            # pad the utterances
            for _ in range(max_len_utterances - len(utterances)):
                utterances_copy.append('')

            new_batch_utterances.append(utterances_copy)

        return {
            'speakers': batch_speakers,
            'emotions': batch_emotions,
            'utterances': new_batch_utterances,
            'triggers': batch_triggers
        }
    
    def encode_utterance(self, t, dialogue, speakers, bert=False):
        # Not used, tokenizer add it automatically
        cls = "[CLS]" if bert else "<s>"
        eos = "[EOS]" if bert else "</s>"
        sep = "[SEP]" if bert else "</s>"

        sequence = f"{sep} {speakers[t].upper()}: {dialogue[t]} {sep}"
        for i in range(1, len(dialogue)):
            # Append next utterance
            if t+i < len(dialogue):
                sequence = sequence + f" {speakers[t+i].upper()}: {dialogue[t+i]} "
            # Append past utterance
            if t-i >= 0:
                sequence = f" {speakers[t-i].upper()}: {dialogue[t-i]} " + sequence
        sequence = sequence
        return sequence

    def collate_context(self, batch):
        padding_value_emotion = 7
        padding_value_trigger = 2
        speakers, emotions, utterances, triggers = zip(*batch)

        emotions = [torch.tensor(e, dtype=torch.long) for e in emotions]
        triggers = [torch.tensor(t, dtype=torch.long) for t in triggers]

        emotions = torch.nn.utils.rnn.pad_sequence(emotions, batch_first=True, padding_value=padding_value_emotion)
        triggers = torch.nn.utils.rnn.pad_sequence(triggers, batch_first=True, padding_value=padding_value_trigger)
        # Pad with a PAD sentence

        max_len_utterances = max([len(u) for u in utterances])

        new_utterances = []
        # For each dialogue
        for i, u in enumerate(utterances):
            # copy the utterances, this is necessary because list are mutable
            dialogue = u.copy()
            for t in range(len(u)):
                sequence = self.encode_utterance(t, u, speakers[i])
                dialogue[t] = sequence
            for _ in range(max_len_utterances - len(u)):
                dialogue.append('')

            new_utterances.append(dialogue)

        return {
            'speakers': speakers,
            'emotions': emotions,
            'utterances': new_utterances,
            'triggers': triggers
        }

    def train_dataloader(self, collate_context=False, batch_size=None):
        if collate_context:
            collate_fn = self.collate_context
        else:
            collate_fn = self.collate
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=self.num_workers)

    def val_dataloader(self, collate_context=False, batch_size=None):
        if collate_context:
            collate_fn = self.collate_context
        else:
            collate_fn = self.collate
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                          num_workers=self.num_workers)

    def test_dataloader(self, collate_context=False, batch_size=None):
        if collate_context:
            collate_fn = self.collate_context
        else:
            collate_fn = self.collate
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                          num_workers=self.num_workers)
