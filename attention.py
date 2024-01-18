import lightning as pl
import numpy as np
import torch

from common_models import CLF, BertEncoder, MetricModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DialoguesAttention(MetricModel):
    def __init__(self, hidden_size=128, emotion_output_dim=7, trigger_output_dim=2, lr=1e-3, freeze_bert=True,
                 class_weights_emotion: torch.Tensor | None = None, class_weights_trigger: torch.Tensor | None = None):
        super().__init__(emotion_output_dim=emotion_output_dim, trigger_output_dim=trigger_output_dim,
                         padding_value_emotion=emotion_output_dim, padding_value_trigger=trigger_output_dim)
        
        self.encoder = BertEncoder('bert-base-uncased', emotion_output_dim, trigger_output_dim, freeze_bert)
        self.lr = lr
        self.class_weights_emotion = class_weights_emotion.to(device)
        self.class_weights_trigger = class_weights_trigger.to(device)

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim

        # Padding value for the emotion and the trigger output 
        self.padding_value_emotion = emotion_output_dim
        self.padding_value_trigger = trigger_output_dim

        self.bert_output_dim = self.encoder.output_dim

        self.attention = torch.nn.MultiheadAttention(self.bert_output_dim, num_heads=2, dropout=0.1, batch_first=True)

        self.emotion_clf = CLF(self.bert_output_dim , hidden_size, self.emotion_output_dim)
        self.trigger_clf = CLF(self.bert_output_dim, hidden_size, self.trigger_output_dim)
        self.save_hyperparameters()


    def forward(self, x):
        batch_utterances = x['utterances']
        batch_encoded_flattened_utterances = [utterance for utterances in batch_utterances for utterance in utterances]
        # Reshape the batch of utterances into a list of utterances
        batch_encoded_utterances = self.encoder.encode(batch_encoded_flattened_utterances)
        batch_encoded_utterances = batch_encoded_utterances.reshape(len(batch_utterances), -1, self.bert_output_dim)

        features = self.attention(batch_encoded_utterances, batch_encoded_utterances, batch_encoded_utterances, need_weights=False)

        # Classification
        emotion_logits = self.emotion_clf(features)
        trigger_logits = self.trigger_clf(features)
        return emotion_logits.to(device), trigger_logits.to(device)

    def on_save_checkpoint(self, checkpoint):
        self.encoder.on_save_checkpoint(checkpoint)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

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

        self.metric_update(type, y_hat_class_emotion, y_emotion, y_hat_class_trigger, y_trigger)

        loss = emotion_loss + trigger_loss
        self.log(f'{type}_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'test')
