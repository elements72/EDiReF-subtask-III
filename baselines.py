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

'''
Random Classifier

'''
class RandomUniformClassifier(pl.LightningModule):
    def __init__(self, num_emotion_classes, num_trigger_classes):
        super().__init__()
        self._random_state = np.random.RandomState()
        self.num_emotion_classes = num_emotion_classes
        self.num_trigger_classes = num_trigger_classes

    def predict_emotions(self, X):
        batch_size = X.size(0)
        #mask to ignore padding in predictions
        mask = X == 7
        emotions_logits = self._random_state.randint(low=0, high=self.num_emotion_classes, size=(batch_size, X.size(1)))
        emotions_logits[mask] = 7
        emotions_predictions = torch.tensor(emotions_logits, dtype=torch.int).to(device)
        return emotions_predictions

    def predict_triggers(self, X):
        batch_size = X.size(0)
        #mask to ignore padding in predictions
        mask = X == 2
        triggers_logits = self._random_state.randint(low=0, high=self.num_trigger_classes, size=(batch_size, X.size(1)))
        triggers_logits[mask] = 2
        triggers_predictions = torch.tensor(triggers_logits > 0.5, dtype=torch.int).to(device)
        return triggers_predictions

'''
Majority Classifier

'''
class MajorityClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.majority_emotions_class = None
        self.majority_triggers_class = None

    def fit(self, train_dataloader):
       
        emotions_list = []
        triggers_list = []
        for batch in train_dataloader:
            emotions = batch["emotions"]
            triggers = batch["triggers"]

        if emotions.size(1) != triggers.size(1):
                raise ValueError("Dimension mismatch along the second dimension.")
        
        emotions_list.append(emotions)
        triggers_list.append(triggers)
        emotions_labels = torch.cat(emotions_list, dim=0)
        triggers_labels = torch.cat(triggers_list, dim=0)

    
        unique_emotions, counts_emotions = torch.unique(emotions_labels, return_counts=True)
        majority_emotions_class = unique_emotions[counts_emotions.argmax()].item()

        unique_triggers, counts_triggers = torch.unique(triggers_labels, return_counts=True)
        majority_triggers_class = unique_triggers[counts_triggers.argmax()].item()

        self.majority_emotions_class = majority_emotions_class
        self.majority_triggers_class = majority_triggers_class
    
    def predict_emotions(self, x):
        return torch.tensor([self.majority_emotions_class] * len(x))

    def predict_triggers(self, x):
        return torch.tensor([self.majority_triggers_class] * len(x))
