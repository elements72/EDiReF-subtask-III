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
        print(checkpoint['state_dict'].keys())

        if self.freeze:
            # Remove all the parameters of the backbone from the checkpoint
            for name, param in self.model.named_parameters():
                del checkpoint['state_dict'][self.encoder_name + ".model." + name]

    def encode(self, x):
        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True).to(device)
        x = self.model(**x).last_hidden_state[:, 0, :]
        return x

    @property
    def output_dim(self):
        return self.model.config.hidden_size


