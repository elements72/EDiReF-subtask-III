import torch

from common_models import BertEncoder, ClassificationTaskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMResModel(ClassificationTaskModel):
    """
                    residual connection ->
        encoder ->                          concat -> classifier
                    lstm                ->
    """

    def __init__(self, lstm_kwargs: dict, emotion_output_dim=7, trigger_output_dim=2, freeze_bert=True,
                 bert_model_name='bert-base-uncased', use_encoder_cache: bool = True, encoder_cache_size: int = 100_000,
                 **kwargs):
        encoder = BertEncoder(bert_model_name, emotion_output_dim, trigger_output_dim, freeze_bert,
                              cache_output=use_encoder_cache, cache_size=encoder_cache_size)
        lstm = torch.nn.LSTM(input_size=encoder.output_dim, **lstm_kwargs, batch_first=True)
        clf_input_dim = encoder.output_dim + (lstm.hidden_size * 2 if lstm_kwargs['bidirectional'] else lstm.hidden_size)

        super().__init__(emotion_output_dim=emotion_output_dim, trigger_output_dim=trigger_output_dim,
                         clf_input_size=clf_input_dim, **kwargs)

        self.encoder = encoder
        self.lstm = lstm

        self.save_hyperparameters()

    def on_save_checkpoint(self, checkpoint):
        self.encoder.on_save_checkpoint(checkpoint)

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
