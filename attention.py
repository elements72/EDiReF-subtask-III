import torch

from common_models import BertEncoder, ClassificationTaskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DialoguesAttention(ClassificationTaskModel):
    def __init__(self, emotion_output_dim=7, trigger_output_dim=2, freeze_bert=True,
                 bert_model_name='bert-base-uncased', use_encoder_cache: bool = True,
                 encoder_cache_size: int = 10_000, **kwargs):
        encoder = BertEncoder(bert_model_name, emotion_output_dim, trigger_output_dim, freeze_bert,
                              cache_output=use_encoder_cache, cache_size=encoder_cache_size)
        attention = torch.nn.MultiheadAttention(encoder.output_dim, num_heads=2, dropout=0.1, batch_first=True)

        super().__init__(emotion_output_dim=emotion_output_dim, trigger_output_dim=trigger_output_dim,
                         clf_input_size=encoder.output_dim * 2, **kwargs)

        self.encoder = encoder
        self.attention = attention

        self.save_hyperparameters()

    def forward(self, x):
        encoded_utterances = self.encoder.encode(x['utterances'])
        attention_out, _ = self.attention(encoded_utterances, encoded_utterances, encoded_utterances)
        clf_input = torch.cat([encoded_utterances, attention_out], dim=-1)
        # Classification
        emotion_logits = self.emotion_clf(clf_input)
        trigger_logits = self.trigger_clf(clf_input)
        return emotion_logits.to(device), trigger_logits.to(device)

    def on_save_checkpoint(self, checkpoint):
        self.encoder.on_save_checkpoint(checkpoint)