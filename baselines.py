import torch
import lightning as pl
from transformers import BertModel, BertTokenizerFast
from metrics import F1ScoreCumulative, F1ScoreDialogues

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, hidden_size=128, emotion_output_dim=7, trigger_output_dim=3, lr=1e-3,
                  freeze_bert=True, padding_value_emotion: int = None, padding_value_trigger: int = None):
        super().__init__()
        self.backbone = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.lr = lr
        self.freeze_bert = freeze_bert

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim

        ## Utterace1, Utterace1, Utterance3, Padding 1 
        ## Trigger 1, Trigger 1, Trigger 1, Padding 1
        ## Emotion 1, Emotion 1, Emotion 1, Padding 1 
        ##

        # Padding value for the emotion and the trigger output 
        self.padding_value_emotion = padding_value_emotion if padding_value_emotion else emotion_output_dim
        self.padding_value_trigger = padding_value_trigger if padding_value_trigger else trigger_output_dim

        self.bert_output_dim = self.backbone.config.hidden_size
        self.emotion_clf = CLF(self.bert_output_dim, hidden_size, self.emotion_output_dim)
        self.trigger_clf = CLF(self.bert_output_dim, hidden_size, self.trigger_output_dim)

        self.f1_train_cumulative_emotion = F1ScoreCumulative(num_classes=self.emotion_output_dim, padding_value=self.padding_value_emotion)
        self.f1_train_cumulative_trigger = F1ScoreCumulative(num_classes=self.trigger_output_dim, padding_value=self.padding_value_trigger)
        self.f1_train_dialogues_emotion = F1ScoreDialogues(num_classes=self.emotion_output_dim, padding_value=self.padding_value_emotion)
        self.f1_train_dialogues_trigger = F1ScoreDialogues(num_classes=self.trigger_output_dim, padding_value=self.padding_value_trigger)

        self.f1_val_cumulative_emotion = F1ScoreCumulative(num_classes=self.emotion_output_dim, padding_value=self.padding_value_emotion)
        self.f1_val_cumulative_trigger = F1ScoreCumulative(num_classes=self.trigger_output_dim, padding_value=self.padding_value_trigger)
        self.f1_val_dialogues_emotion = F1ScoreDialogues(num_classes=self.emotion_output_dim, padding_value=self.padding_value_emotion)
        self.f1_val_dialogues_trigger = F1ScoreDialogues(num_classes=self.trigger_output_dim, padding_value=self.padding_value_trigger)

        self.f1_test_cumulative_emotion = F1ScoreCumulative(num_classes=self.emotion_output_dim, padding_value=self.padding_value_emotion)
        self.f1_test_cumulative_trigger = F1ScoreCumulative(num_classes=self.trigger_output_dim, padding_value=self.padding_value_trigger)
        self.f1_test_dialogues_emotion = F1ScoreDialogues(num_classes=self.emotion_output_dim, padding_value=self.padding_value_emotion)
        self.f1_test_dialogues_trigger = F1ScoreDialogues(num_classes=self.trigger_output_dim, padding_value=self.padding_value_trigger)

        self.f1_cumulative_emotion={
            'train': self.f1_train_cumulative_emotion,
            'val': self.f1_val_cumulative_emotion,
            'test': self.f1_test_cumulative_emotion
        }
        self.f1_cumulative_trigger={
            'train': self.f1_train_cumulative_trigger,
            'val': self.f1_val_cumulative_trigger,
            'test': self.f1_test_cumulative_trigger
        }
        self.f1_dialogues_emotion={
            'train': self.f1_train_dialogues_emotion,
            'val': self.f1_val_dialogues_emotion,
            'test': self.f1_test_dialogues_emotion
        }
        self.f1_dialogues_trigger={
            'train': self.f1_train_dialogues_trigger,
            'val': self.f1_val_dialogues_trigger,
            'test': self.f1_test_dialogues_trigger
        }

        self.save_hyperparameters()
        if freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def on_save_checkpoint(self, checkpoint):
        if self.freeze_bert:
            # Remove all the parameters of the backbone from the checkpoint
            for name, param in self.backbone.named_parameters():
                del checkpoint['state_dict']['backbone.' + name]
    
    def encode(self, x):
        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True).to(self.device)
        x = self.backbone(**x).last_hidden_state[:, 0, :]
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        batch_utterances = x['utterances']
        batch_encoded_flattened_utterances = [utterance for utterances in batch_utterances for utterance in utterances]
        # Reshape the batch of utterances into a list of utterances
        batch_encoded_utterances = self.encode(batch_encoded_flattened_utterances)
        batch_encoded_utterances = batch_encoded_utterances.reshape(len(batch_utterances), -1, self.bert_output_dim)
        # Pad with zeros
        emotion_logits = self.emotion_clf(batch_encoded_utterances)
        trigger_logits = self.trigger_clf(batch_encoded_utterances)
        return emotion_logits.to(device), trigger_logits.to(device)
    
    def type_step(self, batch, batch_idx, type):
        emotion_logits, trigger_logits = self(batch)

        # Since the trigger is a binary classification task, we need to squeeze the last dimension
        # [batch_size, seq_len, 1] -> [batch_size, seq_len] 
        trigger_logits = trigger_logits.squeeze(-1)
        trigger_logits = torch.movedim(trigger_logits, 1, 2)
        emotion_logits = torch.movedim(emotion_logits, 1, 2)

        y_hat_class_emotion = emotion_logits #torch.argmax(emotion_logits, dim=2)
        y_hat_class_trigger = trigger_logits #torch.argmax(trigger_logits, dim=2)

        y_emotion = batch['emotions'].to(device)
        y_trigger = batch['triggers'].to(device)

        emotion_loss = torch.nn.CrossEntropyLoss(ignore_index=self.padding_value_emotion)(emotion_logits, y_emotion)
        trigger_loss = torch.nn.CrossEntropyLoss(ignore_index=self.padding_value_trigger)(trigger_logits, y_trigger)

        self.f1_cumulative_emotion[type].update(y_hat_class_emotion, y_emotion)
        self.f1_cumulative_trigger[type].update(y_hat_class_trigger, y_trigger)

        self.f1_dialogues_emotion[type].update(y_hat_class_emotion, y_emotion)
        self.f1_dialogues_trigger[type].update(y_hat_class_trigger, y_trigger)

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
            f'f1_{type}_cumulative_emotion': self.f1_cumulative_emotion[type].compute(),
            f'f1_{type}_cumulative_trigger': self.f1_cumulative_trigger[type].compute(),
            f'f1_{type}_dialogues_emotion': self.f1_dialogues_emotion[type].compute(),
            f'f1_{type}_dialogues_trigger': self.f1_dialogues_trigger[type].compute()
            }, prog_bar=True, on_epoch=True, on_step=False)
        self.f1_cumulative_emotion[type].reset()
        self.f1_cumulative_trigger[type].reset()
        self.f1_dialogues_emotion[type].reset()
        self.f1_dialogues_trigger[type].reset()
        
    def on_train_epoch_end(self):
        self.on_epoch_type_end('train')
    def on_validation_epoch_end(self):
        self.on_epoch_type_end('val')
    def on_test_epoch_end(self):
        self.on_epoch_type_end('test')
