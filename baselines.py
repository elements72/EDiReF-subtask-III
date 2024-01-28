import lightning as pl
import numpy as np
import torch
from metrics import F1ScoreCumulative, F1ScoreDialogues
from common_models import BertEncoder, ClassificationTaskModel
import pandas as pd
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
RANDOM CLASSIFIER

'''
class RandomUniformClassifier(pl.LightningModule):
    def __init__(self, num_emotion_classes, num_trigger_classes):
        super().__init__()
        self._random_state = np.random.RandomState()
        self.num_emotion_classes = num_emotion_classes
        self.num_trigger_classes = num_trigger_classes

    def predict_emotions(self, X):
        batch_size = X.size(0)
        emotions_logits = self._random_state.randint(low=0, high=self.num_emotion_classes, size=(batch_size, X.size(1)))
        emotions_predictions = torch.tensor(emotions_logits, dtype=torch.int).to(device)
        return emotions_predictions

    def predict_triggers(self, X):
        batch_size = X.size(0)
        triggers_logits = self._random_state.randint(low=0, high=self.num_trigger_classes, size=(batch_size, X.size(1)))
        triggers_predictions = torch.tensor(triggers_logits, dtype=torch.int).to(device)
        return triggers_predictions

'''

Predictions of the Random Classifier and calculation of metrics

'''
def random_metrics(random, test_loader, num_classes_emotions, num_classes_triggers):

    #Metrics
    emotions_f1_cumulative = F1ScoreCumulative(num_classes=num_classes_emotions, binary=False, padding_value=7)
    emotions_f1_dialogues = F1ScoreDialogues(num_classes=num_classes_emotions, binary=False, padding_value=7)

    triggers_f1_cumulative_binary = F1ScoreCumulative(num_classes=num_classes_triggers, binary=True, padding_value=2)
    triggers_f1_dialogues_binary = F1ScoreDialogues(num_classes=num_classes_triggers, binary=True, padding_value=2)

    triggers_f1_cumulative_multiclass = F1ScoreCumulative(num_classes=num_classes_triggers, binary=False, padding_value=2)
    triggers_f1_dialogues_multiclass = F1ScoreDialogues(num_classes=num_classes_triggers, binary=False, padding_value=2)

    #Computing Random predictions
    emotions_predictions = []
    triggers_predictions = []


    for batch in test_loader:
        emotions = batch["emotions"]
        triggers = batch["triggers"]

        # Emotions prediction
        predictions_emotions = random.predict_emotions(emotions)
        emotions_predictions.append(predictions_emotions)
        # Triggers prediction 
        predictions_trigger = random.predict_triggers(triggers)
        triggers_predictions.append(predictions_trigger)

        y_hat_class_emotions = predictions_emotions
        y_hat_class_trigger_binary = predictions_trigger
        y_hat_class_trigger_multiclass = predictions_trigger
        y_class_emotions = batch["emotions"]
        y_class_trigger = batch["triggers"]
        
    
        y_hat_class_trigger_binary = y_hat_class_trigger_binary 
        y_class_trigger_binary = y_class_trigger 
        # Emotions metrics
        emotions_f1_cumulative.update(y_hat_class_emotions, y_class_emotions)
        emotions_f1_dialogues.update(y_hat_class_emotions, y_class_emotions)

        # Triggers metrics (binary=True)
        triggers_f1_cumulative_binary.update(y_hat_class_trigger_binary, y_class_trigger_binary)
        triggers_f1_dialogues_binary.update(y_hat_class_trigger_binary, y_class_trigger_binary)

        # Triggers metrics (binary=False)
        triggers_f1_cumulative_multiclass.update(y_hat_class_trigger_multiclass, y_class_trigger)
        triggers_f1_dialogues_multiclass.update(y_hat_class_trigger_multiclass, y_class_trigger)

    # Concatenate the predictions
    emotions_predictions = torch.cat(emotions_predictions, dim=1)
    triggers_predictions = torch.cat(triggers_predictions, dim=1)

    # Compute F1 Scores...
    #Emotions
    emotions_f1_cumulative_result = emotions_f1_cumulative.compute().item()
    emotions_f1_dialogues_result = emotions_f1_dialogues.compute().item()
    #Triggers (Binary)
    triggers_f1_cumulative_binary_result = triggers_f1_cumulative_binary.compute().item()
    triggers_f1_dialogues_binary_result = triggers_f1_dialogues_binary.compute().item()
    #Triggers (Multiclass)
    triggers_f1_cumulative_multiclass_result = triggers_f1_cumulative_multiclass.compute().item()
    triggers_f1_dialogues_multiclass_result = triggers_f1_dialogues_multiclass.compute().item()

    # Save Results
    metrics_data = {
    "Emotions": [emotions_f1_cumulative_result, emotions_f1_dialogues_result],
    "Triggers Binary": [triggers_f1_cumulative_binary_result, triggers_f1_dialogues_binary_result],
    "Triggers Multiclass": [triggers_f1_cumulative_multiclass_result, triggers_f1_dialogues_multiclass_result],
    }   

    metrics_df = pd.DataFrame(metrics_data, index=["F1 Cumulative", "F1 Dialogues"])

    return metrics_df, emotions_predictions, triggers_predictions


'''
MAJORITY CLASSIFIER

'''
class MajorityClassifier(pl.LightningModule):
    def __init__(self, num_classes_emotions, num_classes_triggers, padding_value_emotions, padding_value_triggers):
        super().__init__()
        self.majority_emotions_class = None
        self.majority_triggers_class = None
        self.num_classes_emotions = num_classes_emotions
        self.num_classes_triggers = num_classes_triggers
        self.padding_value_emotions = padding_value_emotions
        self.padding_value_triggers = padding_value_triggers

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

        emotions_labels = torch.cat(emotions_list, dim=1)
        triggers_labels = torch.cat(triggers_list, dim=1)

        unique_emotions, counts_emotions = torch.unique(emotions_labels[emotions_labels != 7], return_counts=True)
        majority_emotions_class = unique_emotions[counts_emotions.argmax()].item() 
        unique_triggers, counts_triggers = torch.unique(triggers_labels[triggers_labels != 2], return_counts=True)
        majority_triggers_class = unique_triggers[counts_triggers.argmax()].item()
        self.majority_emotions_class = majority_emotions_class
        self.majority_triggers_class = majority_triggers_class
    
    def predict_emotions(self, x):
        predictions = torch.full_like(x, fill_value=self.majority_emotions_class)
        
        return predictions


    def predict_triggers(self, x):
        predictions = torch.full_like(x, fill_value=self.majority_triggers_class)
        
        
        return predictions

'''
Predictions of the Majority Classifier and calculation of metrics

'''

def majority_metrics(majority_classifier, test_loader, num_classes_emotions, num_classes_triggers):

    #Metrics
    emotions_f1_cumulative = F1ScoreCumulative(num_classes=num_classes_emotions, binary=False, padding_value=7)
    emotions_f1_dialogues = F1ScoreDialogues(num_classes=num_classes_emotions, binary=False, padding_value=7)

    triggers_f1_cumulative_binary = F1ScoreCumulative(num_classes=num_classes_triggers, binary=True, padding_value=2)
    triggers_f1_dialogues_binary = F1ScoreDialogues(num_classes=num_classes_triggers, binary=True, padding_value=2)

    triggers_f1_cumulative_multiclass = F1ScoreCumulative(num_classes=num_classes_triggers, binary=False, padding_value=2)
    triggers_f1_dialogues_multiclass = F1ScoreDialogues(num_classes=num_classes_triggers, binary=False, padding_value=2)

    # Computing Majority predictions
    emotions_predictions = []
    triggers_predictions = []

    for batch in test_loader:
        emotions = batch["emotions"]
        triggers = batch["triggers"]

        # Emotions prediction
        predictions_emotions = majority_classifier.predict_emotions(emotions)
        emotions_predictions.append(predictions_emotions)
        # Triggers prediction
        predictions_trigger = majority_classifier.predict_triggers(triggers)
        triggers_predictions.append(predictions_trigger)
       

        y_hat_class_emotions = predictions_emotions
        y_hat_class_trigger_binary = predictions_trigger
        y_hat_class_trigger_multiclass = predictions_trigger
        y_class_emotions = batch["emotions"]
        y_class_trigger = batch["triggers"]
        # Mask padding

        y_hat_class_trigger_binary = y_hat_class_trigger_binary
        y_class_trigger_binary = y_class_trigger
        # Emotions metrics
        emotions_f1_cumulative.update(y_hat_class_emotions, y_class_emotions)
        emotions_f1_dialogues.update(y_hat_class_emotions, y_class_emotions)

        # Triggers metrics (binary=True)
        triggers_f1_cumulative_binary.update(y_hat_class_trigger_binary, y_class_trigger_binary)
        triggers_f1_dialogues_binary.update(y_hat_class_trigger_binary, y_class_trigger_binary)

        # Triggers metrics (binary=False)
        triggers_f1_cumulative_multiclass.update(y_hat_class_trigger_multiclass, y_class_trigger)
        triggers_f1_dialogues_multiclass.update(y_hat_class_trigger_multiclass, y_class_trigger)

    # Concatenate the predictions
    emotions_predictions = torch.cat(emotions_predictions, dim=1)
    triggers_predictions = torch.cat(triggers_predictions, dim=1)

    # Compute F1 Scores...
    # Emotions
    emotions_f1_cumulative_result = emotions_f1_cumulative.compute().item()
    emotions_f1_dialogues_result = emotions_f1_dialogues.compute().item()
    # Triggers (Binary)
    triggers_f1_cumulative_binary_result = triggers_f1_cumulative_binary.compute().item()
    triggers_f1_dialogues_binary_result = triggers_f1_dialogues_binary.compute().item()
    
    # Triggers (Multiclass)
    triggers_f1_cumulative_multiclass_result = triggers_f1_cumulative_multiclass.compute().item()
    triggers_f1_dialogues_multiclass_result = triggers_f1_dialogues_multiclass.compute().item()

    # Save Results
    metrics_data = {
        "Emotions": [emotions_f1_cumulative_result, emotions_f1_dialogues_result],
        "Triggers Binary": [triggers_f1_cumulative_binary_result, triggers_f1_dialogues_binary_result],
        "Triggers Multiclass": [triggers_f1_cumulative_multiclass_result, triggers_f1_dialogues_multiclass_result],
    }

    metrics_df = pd.DataFrame(metrics_data, index=["F1 Cumulative", "F1 Dialogues"])

    return metrics_df, emotions_predictions, triggers_predictions

