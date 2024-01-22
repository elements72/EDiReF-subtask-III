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

    def determine_num_classes(self, dataloader, key):
        sample_batch = next(iter(dataloader))
        num_classes = sample_batch[key].size(1)
        return num_classes

    def initialize_classes(self, train_dataloader):
        self.num_emotion_classes = self.determine_num_classes(train_dataloader, "emotions")
        self.num_trigger_classes = self.determine_num_classes(train_dataloader, "triggers")

    def predict_emotions(self, X):
        batch_size = X.size(0)
        emotions_logits = self._random_state.uniform(size=(batch_size, self.num_emotion_classes))
        emotions_predictions = torch.tensor(emotions_logits > 0.5, dtype=torch.float32).to(device)
        return emotions_predictions

    def predict_triggers(self, X):
        batch_size = X.size(0)
        triggers_logits = self._random_state.uniform(size=(batch_size, self.num_trigger_classes))
        triggers_predictions = torch.tensor(triggers_logits > 0.5, dtype=torch.float32).to(device)
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
