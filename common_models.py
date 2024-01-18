import lightning as pl
import torch
from transformers import BertModel, BertTokenizerFast

from metrics import F1ScoreCumulative, F1ScoreDialogues

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLF(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, output_dim=7, lr=1e-3):
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


class BertEncoder(pl.LightningModule):
    def __init__(self, bert_model_name: str, emotion_output_dim: int = 7, trigger_output_dim: int = 2,
                 freeze_bert: bool = True, encoder_name: str = "encoder"):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name, batched=True)

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim

        self.freeze = freeze_bert
        self.encoder_name = encoder_name

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        if self.freeze:
            # Remove all the parameters of the backbone from the checkpoint
            for name, param in self.model.named_parameters():
                del checkpoint['state_dict'][self.encoder_name + ".model." + name]

    def encode(self, utterances):
        flattend = [u for sub_list in utterances for u in sub_list]
        x = self.tokenizer(flattend, return_tensors='pt', padding=True, truncation=True).to(device)
        x = self.model(**x).last_hidden_state[:, 0, :]

        # Reshape the batch of utterances into a list of utterances
        return x.reshape(len(utterances), -1, self.model.config.hidden_size)

    @property
    def output_dim(self):
        return self.model.config.hidden_size


class MetricModel(pl.LightningModule):
    """
    Class for the metric. It contains the metrics for the emotion and the trigger tasks.
    """

    def __init__(self, emotion_output_dim=7, trigger_output_dim=2, padding_value_emotion=7, padding_value_trigger=2):
        super().__init__()

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim
        self.padding_value_emotion = padding_value_emotion
        self.padding_value_trigger = padding_value_trigger

        self.f1_cumulative_emotion = {}
        self.f1_cumulative_trigger = {}
        self.f1_dialogues_emotion = {}
        self.f1_dialogues_trigger = {}
        for stage in ['train', 'val', 'test']:
            self.f1_cumulative_emotion[stage] = F1ScoreCumulative(num_classes=self.emotion_output_dim,
                                                                  padding_value=self.padding_value_emotion).to(device)
            self.f1_cumulative_trigger[stage] = F1ScoreCumulative(num_classes=self.trigger_output_dim,
                                                                  padding_value=self.padding_value_trigger,
                                                                  binary=True).to(device)
            self.f1_dialogues_emotion[stage] = F1ScoreDialogues(num_classes=self.emotion_output_dim,
                                                                padding_value=self.padding_value_emotion).to(device)
            self.f1_dialogues_trigger[stage] = F1ScoreDialogues(num_classes=self.trigger_output_dim,
                                                                padding_value=self.padding_value_trigger,
                                                                binary=True).to(device)

    def metric_update(self, stage: str, y_hat_class_emotion, y_emotion, y_hat_class_trigger, y_trigger):
        self.f1_cumulative_emotion[stage](y_hat_class_emotion, y_emotion)
        self.f1_cumulative_trigger[stage](y_hat_class_trigger, y_trigger)

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
