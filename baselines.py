import torch
import lightning as pl
from transformers import BertModel, BertTokenizerFast


class CLF(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6, lr=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.save_hyperparameters()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class BertBaseline(pl.LightningModule):
    def __init__(self, hidden_size=128, emotion_output_dim=6, lr=1e-3, freeze_bert=False):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.lr = lr

        self.bert_output_dim = self.model.config.hidden_size
        self.emotion_clf = CLF(self.bert_output_dim, hidden_size, emotion_output_dim)
        self.trigger_clf = CLF(self.bert_output_dim, hidden_size, 1)

        self.save_hyperparameters()
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def encode(self, x):
        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        x = self.model(**x).last_hidden_state[:, 0, :]
        return x

    def forward(self, x):
        utterances = x['utterances']
        encoded_utterances = torch.stack([self.encode(utterance) for utterance in utterances], dim=1)
        emotion_logits = self.emotion_clf(encoded_utterances)
        trigger_logits = self.trigger_clf(encoded_utterances)
        return emotion_logits, trigger_logits
    
    def training_step(self, batch, batch_idx):
        emotion_logits, trigger_logits = self(batch)
        emotion_loss = torch.nn.CrossEntropyLoss()(emotion_logits, batch['emotion'])
        trigger_loss = torch.nn.BCELoss()(trigger_logits, batch['trigger'])
        loss = emotion_loss + trigger_loss
        self.log('train_loss', loss)
        return loss