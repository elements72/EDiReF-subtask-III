import torch
import lightning as pl
from transformers import BertModel, BertTokenizerFast
from metrics import F1ScoreCumulative, F1ScoreDialogues

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
    def __init__(self, hidden_size=128, emotion_output_dim=7, trigger_output_dim=3, lr=1e-3, freeze_bert=False):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.lr = lr

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim

        self.bert_output_dim = self.model.config.hidden_size
        self.emotion_clf = CLF(self.bert_output_dim, hidden_size, self.emotion_output_dim)
        self.trigger_clf = CLF(self.bert_output_dim, hidden_size, 3)

        self.f1_cumulative_emotion = F1ScoreCumulative(num_classes=self.emotion_output_dim)
        self.f1_cumulative_triggers = F1ScoreCumulative(num_classes=self.trigger_output_dim)

        self.f1_dialogues_emotion = F1ScoreDialogues(num_classes=self.emotion_output_dim)
        self.f1_dialogues_trigger = F1ScoreDialogues(num_classes=self.trigger_output_dim)

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
    
    def type_step(self, batch, batch_idx, type):
        emotion_logits, trigger_logits = self(batch)

        # Since the trigger is a binary classification task, we need to squeeze the last dimension
        # [batch_size, seq_len, 1] -> [batch_size, seq_len] 
        trigger_logits = trigger_logits.squeeze(-1)
        trigger_logits = torch.movedim(trigger_logits, 1, 2)
        emotion_logits = torch.movedim(emotion_logits, 1, 2)

        emotion_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(emotion_logits, batch['emotions'])
        trigger_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(trigger_logits, batch['triggers'])

        self.f1_cumulative_emotion.update(emotion_logits, batch['emotions'])
        self.f1_cumulative_trigger.update(trigger_logits, batch['triggers'])

        self.f1_dialogues_emotion.update(emotion_logits, batch['emotions'])
        self.f1_dialogues_trigger.update(trigger_logits, batch['triggers'])

        loss = emotion_loss + trigger_loss
        self.log(f'{type}_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'test')
    
    def on_epoch_type_end(self, type):
        self.log_dict({
            f'f1_{type}_cumulative_emotion': self.f1_cumulative.compute(),
            f'f1_{type}_cumulative_trigger': self.f1_cumulative.compute(),
            f'f1_{type}_dialogues_emotion': self.f1_dialogues.compute(),
            f'f1_{type}_dialogues_trigger': self.f1_dialogues.compute()
            }, prog_bar=True, on_epoch=True, on_step=False)
        
    def on_train_epoch_end(self):
        self.on_epoch_type_end('train')
    def on_validation_epoch_end(self):
        self.on_epoch_type_end('val')
    def on_test_epoch_end(self):
        self.on_epoch_type_end('test')
