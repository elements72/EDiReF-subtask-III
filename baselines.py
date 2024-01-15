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
        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True).to(self.device)
        x = self.model(**x).last_hidden_state[:, 0, :]
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        batch_utterances = x['utterances']
        batch_encoded_utterances = []
        for utterances in batch_utterances:
            encoded_utterances = self.encode(utterances)
            batch_encoded_utterances.append(encoded_utterances)
        # Pad with zeros
        batch_encoded_utterances = torch.nn.utils.rnn.pad_sequence(batch_encoded_utterances, batch_first=True, padding_value=-1)
        emotion_logits = self.emotion_clf(batch_encoded_utterances)
        trigger_logits = self.trigger_clf(batch_encoded_utterances)
        return emotion_logits, trigger_logits
    
    def training_step(self, batch, batch_idx):
        emotion_logits, trigger_logits = self(batch)

        # Since the trigger is a binary classification task, we need to squeeze the last dimension
        # [batch_size, seq_len, 1] -> [batch_size, seq_len] 
        trigger_logits = trigger_logits.squeeze(-1)
        emotion_logits = torch.movedim(emotion_logits, 1, 2)

        emotion_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(emotion_logits, batch['emotions'])
        trigger_loss = torch.nn.BCELoss(ignore_index=-1)(trigger_logits, batch['triggers'])
        loss = emotion_loss + trigger_loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        emotion_logits, trigger_logits = self(batch)

        # Since the trigger is a binary classification task, we need to squeeze the last dimension
        # [batch_size, seq_len, 1] -> [batch_size, seq_len] 
        trigger_logits = trigger_logits.squeeze(-1)
        print(emotion_logits.shape)
        print(batch['emotions'].shape)
        print(trigger_logits.shape)
        print(batch['triggers'].shape)

        emotion_logits = torch.movedim(emotion_logits, 1, 2)

        emotion_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(emotion_logits, batch['emotions'])
        trigger_loss = torch.nn.BCELoss(ignore_index=-1)(trigger_logits, batch['triggers'])
        loss = emotion_loss + trigger_loss
        self.log('val_loss', loss)
        return loss