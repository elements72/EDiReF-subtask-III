import lightning as pl
import numpy as np
import torch

from common_models import BertEncoder, ClassificationTaskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertBaseline(ClassificationTaskModel):
    def __init__(self, emotion_output_dim=7, trigger_output_dim=2, freeze_bert=True,
                 bert_model_name='bert-base-uncased', use_encoder_cache: bool = True,
                 encoder_cache_size: int = 10_000, **kwargs):
        encoder = BertEncoder(bert_model_name, emotion_output_dim, trigger_output_dim, freeze_bert,
                              cache_output=use_encoder_cache, cache_size=encoder_cache_size)

        super().__init__(emotion_output_dim=emotion_output_dim, trigger_output_dim=trigger_output_dim,
                         clf_input_size=encoder.output_dim, **kwargs)

        self.encoder = encoder
        self.save_hyperparameters()

    def on_save_checkpoint(self, checkpoint):
        self.encoder.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.encoder.on_load_checkpoint(checkpoint)

    def forward(self, x):
        encoded_utterances = self.encoder.encode(x['utterances'])
        emotion_logits = self.emotion_clf(encoded_utterances)
        trigger_logits = self.trigger_clf(encoded_utterances)
        return emotion_logits.to(device), trigger_logits.to(device)


class RandomUniformClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._random_state = np.random.RandomState()

    def predict(self, X):
        batch_size = X.size(0)
        logits = self._random_state.uniform(size=(batch_size, 6))
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
