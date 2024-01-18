import torch

from common_models import CLF, BertEncoder, MetricModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMResModel(MetricModel):
    """
                    residual connection ->
        encoder ->                          concat -> classifier
                    lstm                ->
    """

    def __init__(
            self,
            lstm_kwargs: dict,
            hidden_size=128,
            emotion_output_dim=7, trigger_output_dim=2,
            lr=1e-3, freeze_bert=True,
            class_weights_emotion: torch.Tensor | None = None, class_weights_trigger: torch.Tensor | None = None

    ):
        super().__init__(emotion_output_dim=emotion_output_dim, trigger_output_dim=trigger_output_dim,
                         padding_value_emotion=emotion_output_dim, padding_value_trigger=trigger_output_dim, )

        self.encoder = BertEncoder('bert-base-uncased', emotion_output_dim, trigger_output_dim, freeze_bert)

        self.lr = lr
        self.class_weights_emotion = class_weights_emotion.to(device)
        self.class_weights_trigger = class_weights_trigger.to(device)

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim

        print(f"Encoder output dim: {self.encoder.output_dim}")

        print(f"lstm_kwargs: {lstm_kwargs}")

        self.lstm = torch.nn.LSTM(input_size=self.encoder.output_dim, **lstm_kwargs)

        # Padding value for the emotion and the trigger output
        self.padding_value_emotion = emotion_output_dim
        self.padding_value_trigger = trigger_output_dim

        clf_input_dim = self.encoder.output_dim + self.lstm.hidden_size

        print(f"clf_input_dim: {clf_input_dim}")

        self.emotion_clf = CLF(clf_input_dim, hidden_size, self.emotion_output_dim)
        self.trigger_clf = CLF(clf_input_dim, hidden_size, self.trigger_output_dim)

        self.save_hyperparameters()

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

    def forward(self, x):
        encoded_utterances = self.encoder.encode(x['utterances'])

        # Run it through the LSTM
        lstm_out, _ = self.lstm(encoded_utterances)
        lstm_out = lstm_out[:, -1, :]

        # lstm_out shape is [batch_size, hidden_size]
        # encoded_utterances shape is [batch_size, seq_len, bert_output_dim]
        # i need to concat the two tensors along the seq_len dimension
        # so that the final shape is [batch_size, seq_len, bert_output_dim + hidden_size]

        seq_len = encoded_utterances.shape[1]
        lstm_out = lstm_out.unsqueeze(1).repeat(1, seq_len, 1)

        clf_input = torch.cat([encoded_utterances, lstm_out], dim=-1)

        # Run it through the classifier
        emotion_logits = self.emotion_clf(clf_input)
        trigger_logits = self.trigger_clf(clf_input)
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
