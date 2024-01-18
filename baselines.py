import lightning as pl
import numpy as np
import torch

from common_models import CLF, BertEncoder
from metrics import F1ScoreCumulative, F1ScoreDialogues

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertBaseline(pl.LightningModule):
    def __init__(self, hidden_size=128, emotion_output_dim=7, trigger_output_dim=2, lr=1e-3, freeze_bert=True,
                 class_weights_emotion: torch.Tensor | None = None, class_weights_trigger: torch.Tensor | None = None):
        super().__init__()

        self.encoder = BertEncoder('bert-base-uncased', emotion_output_dim, trigger_output_dim, freeze_bert)

        self.lr = lr
        self.class_weights_emotion = class_weights_emotion
        self.class_weights_trigger = class_weights_trigger

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim

        # Padding value for the emotion and the trigger output 
        self.padding_value_emotion = emotion_output_dim
        self.padding_value_trigger = trigger_output_dim

        self.bert_output_dim = self.encoder.output_dim
        self.emotion_clf = CLF(self.bert_output_dim, hidden_size, self.emotion_output_dim)
        self.trigger_clf = CLF(self.bert_output_dim, hidden_size, self.trigger_output_dim)

        self.f1_cumulative_emotion = {}
        self.f1_cumulative_trigger = {}
        self.f1_dialogues_emotion = {}
        self.f1_dialogues_trigger = {}
        for stage in ['train', 'val', 'test']:
            self.f1_cumulative_emotion[stage] = F1ScoreCumulative(num_classes=self.emotion_output_dim,
                                                                  padding_value=self.padding_value_emotion)
            self.f1_cumulative_trigger[stage] = F1ScoreCumulative(num_classes=self.trigger_output_dim,
                                                                  padding_value=self.padding_value_trigger, binary=True)
            self.f1_dialogues_emotion[stage] = F1ScoreDialogues(num_classes=self.emotion_output_dim,
                                                                padding_value=self.padding_value_emotion)
            self.f1_dialogues_trigger[stage] = F1ScoreDialogues(num_classes=self.trigger_output_dim,
                                                                padding_value=self.padding_value_trigger, binary=True)

        self.save_hyperparameters()

    def on_save_checkpoint(self, checkpoint):
        self.encoder.on_save_checkpoint(checkpoint)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        batch_utterances = x['utterances']
        batch_encoded_flattened_utterances = [utterance for utterances in batch_utterances for utterance in utterances]
        # Reshape the batch of utterances into a list of utterances
        batch_encoded_utterances = self.encoder.encode(batch_encoded_flattened_utterances)
        batch_encoded_utterances = batch_encoded_utterances.reshape(len(batch_utterances), -1, self.bert_output_dim)
        # Pad with zeros
        emotion_logits = self.emotion_clf(batch_encoded_utterances)
        trigger_logits = self.trigger_clf(batch_encoded_utterances)
        return emotion_logits.to(device), trigger_logits.to(device)

    def type_step(self, batch, batch_idx, type):
        emotion_logits, trigger_logits = self(batch)

        batch_size = emotion_logits.size(0)
        # Since the trigger is a binary classification task, we need to squeeze the last dimension
        # [batch_size, seq_len, 1] -> [batch_size, seq_len] 
        trigger_logits = trigger_logits.squeeze(-1)
        trigger_logits = torch.movedim(trigger_logits, 1, 2)
        emotion_logits = torch.movedim(emotion_logits, 1, 2)

        y_hat_class_emotion = emotion_logits  # torch.argmax(emotion_logits, dim=2)
        y_hat_class_trigger = trigger_logits  # torch.argmax(trigger_logits, dim=2)

        y_emotion = batch['emotions'].to(device)
        y_trigger = batch['triggers'].to(device)

        emotion_loss_obj = torch.nn.CrossEntropyLoss(ignore_index=self.padding_value_emotion,
                                                     weight=self.class_weights_emotion)
        trigger_loss_obj = torch.nn.CrossEntropyLoss(ignore_index=self.padding_value_trigger,
                                                     weight=self.class_weights_trigger)

        emotion_loss = emotion_loss_obj(emotion_logits, y_emotion)
        trigger_loss = trigger_loss_obj(trigger_logits, y_trigger)

        self.f1_cumulative_emotion[type].update(y_hat_class_emotion, y_emotion)
        self.f1_cumulative_trigger[type].update(y_hat_class_trigger, y_trigger)

        self.f1_dialogues_emotion[type].update(y_hat_class_emotion, y_emotion)
        self.f1_dialogues_trigger[type].update(y_hat_class_trigger, y_trigger)

        loss = emotion_loss + trigger_loss
        self.log(f'{type}_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
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


class RandomUniformClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._random_state = np.random.RandomState()

    def predict(self, X):
        batch_size = X.size(0)
        logits = self._random_state.uniform(size=(batch_size, 4))
        logits = logits > 0.5
        return torch.tensor(logits, dtype=torch.float32).to(device)


class MajorityClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def fit(self, train_dataloader):
        labels = torch.cat([batch["labels"] for batch in train_dataloader])
        majority_class = labels.mode()[0].item()
        self.majority_class = torch.tensor([1 if i == majority_class else 0 for i in range(self.num_classes)])

    def predict(self, x):
        batch_size = x.size(0)
        return self.majority_class.repeat(batch_size, 1)
