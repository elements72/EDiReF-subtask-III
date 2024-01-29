import lightning as pl
import torch
from transformers import BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizerFast

from metrics import F1ScoreCumulative, F1ScoreDialogues
from utils import FIFOCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLF(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, output_dim=7, hidden_layers=1, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.save_hyperparameters()

        setattr(self, f'fc0', torch.nn.Linear(self.input_dim, self.hidden_dim))
        setattr(self, f'relu0', torch.nn.ReLU())
        setattr(self, f'dropout0', torch.nn.Dropout(self.dropout))

        for i in range(1, self.hidden_layers):
            setattr(self, f'fc{i}', torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            setattr(self, f'relu{i}', torch.nn.ReLU())
            setattr(self, f'dropout{i}', torch.nn.Dropout(self.dropout))
        self.out = torch.nn.Linear(self.hidden_dim, self.output_dim)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for i in range(self.hidden_layers):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'relu{i}')(x)
            x = getattr(self, f'dropout{i}')(x)
        x = self.out(x)
        return x


class BertEncoder(pl.LightningModule):
    def __init__(self, bert_model_name: str, emotion_output_dim: int = 7, trigger_output_dim: int = 2,
                 freeze_bert: bool = True, encoder_name: str = "encoder", cache_output: bool = False,
                 cache_size: int = 100_000):
        super().__init__()
        if 'roberta' in bert_model_name:
            self.model = RobertaModel.from_pretrained(bert_model_name)
            self.tokenizer = RobertaTokenizerFast.from_pretrained(bert_model_name, batched=True)
        elif 'emoberta' in bert_model_name:
            self.model = RobertaModel.from_pretrained(bert_model_name)
            self.tokenizer = RobertaTokenizerFast.from_pretrained(bert_model_name, batched=True)
        else:
            self.model = BertModel.from_pretrained(bert_model_name)
            self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name, batched=True)

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim

        self.freeze = freeze_bert
        self.encoder_name = encoder_name

        self.cache_output = cache_output
        self.cache_size = cache_size

        if not self.freeze and self.cache_output:
            raise ValueError("Cannot cache the output if the encoder is not frozen")

        if self.cache_output:
            self.cache: FIFOCache[str, torch.Tensor] = FIFOCache(self.cache_size)
            print(f"Using cache of size {self.cache_size} for the embedding")

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Add a batch norm layer after the bert model
            self.batch_norm = torch.nn.BatchNorm1d(self.model.config.hidden_size)

    def on_save_checkpoint(self, checkpoint):
        if self.freeze:
            # Remove all the parameters of the backbone from the checkpoint
            for name, param in self.model.named_parameters():
                del checkpoint['state_dict'][self.encoder_name + ".model." + name]

    def on_load_checkpoint(self, checkpoint):
        if self.freeze:
            for name, param in self.model.named_parameters():
                checkpoint['state_dict'][self.encoder_name + ".model." + name] = param
        else:
            super().on_load_checkpoint(checkpoint)

    def _encode_with_cache(self, flattened_utterances: list[str]) -> torch.Tensor:
        """
        Encode the utterances using, if possible, the cache.
        If the utterance is not in the cache, it is encoded and added to the cache.
        For the padding a constant embedding of 0 is used.
        """

        # We take the indices of all the not None utterances, so that we avoid computing the
        # embeddings of the None utterances (that are padding)
        non_padding_utterances = [x for x in flattened_utterances if x != ""]

        # From the non padding utterances we keep two indices one for the utterances that are in
        # the cache and one for the utterances that are not in the cache and are to be encoded
        in_cache_indexes = []
        to_encoding_indexes = []

        # We also keep the values of the utterances that are in the cache, so that we can use them later
        in_cache_values = []

        for i, sentence in enumerate(flattened_utterances):
            if sentence == "":
                continue
            cached: torch.Tensor = self.cache.get(sentence)
            if cached is not None:
                in_cache_indexes.append(i)
                in_cache_values.append(cached)
            else:
                to_encoding_indexes.append(i)

        # The utterances that are to be encoded are the ones that are not in the cache
        to_encoding_utterances = [flattened_utterances[i] for i in to_encoding_indexes]

        if len(in_cache_indexes) == 0:
            # If no utterances is in the cache, we encode all the utterances,
            # we keep track of both encoder_out and out (that are the same in this case)
            # so that we can later update the cache
            encoder_out = self._encode_no_cache(flattened_utterances)
            out = encoder_out
        else:
            # If there are utterances in the cache, we encode only the utterances that are not in the cache

            in_cache_values = torch.stack(in_cache_values)
            out = torch.zeros((len(flattened_utterances), self.model.config.hidden_size)).to(device)

            if len(to_encoding_indexes) != 0:
                # If no utterances need to be encoded, we skip this step
                tokenized = self.tokenizer(to_encoding_utterances, return_tensors='pt', padding=True,
                                           truncation=True).to(device)
                encoder_out = self.model(**tokenized).last_hidden_state[:, 0, :]
                out[to_encoding_indexes] = encoder_out

            out[in_cache_indexes] = in_cache_values

        # For all the utterances that were to be encoded, we update the cache
        for i, sentences in enumerate(to_encoding_utterances):
            sentence_encoding = encoder_out[i]
            self.cache.put(sentences, sentence_encoding)

        return out

    def _encode_no_cache(self, flattened_utterances: list[str]) -> torch.Tensor:
        """
        Encode the utterances without using the cache. For the padding a constant embedding of 0 is used, so
        the bert model is not called for the padding.
        """

        # We take the indices of all the padding utterances, so that we avoid computing the
        # embeddings of them
        non_padding_utterances = [x for x in flattened_utterances if x != ""]
        non_padding_indexes = [i for i, x in enumerate(flattened_utterances) if x != ""]

        # We tokenize the non padding utterances
        tokenized = self.tokenizer(non_padding_utterances, return_tensors='pt', padding=True, truncation=True).to(
            device)
        out = torch.zeros((len(flattened_utterances), self.model.config.hidden_size)).to(device)

        # Add to the zero tensor the embeddings of the non padding utterances
        out[non_padding_indexes] = self.model(**tokenized).last_hidden_state[:, 0, :]
        return out

    def encode(self, utterances) -> torch.Tensor:
        flattened_utterances = [u for sub_list in utterances for u in sub_list]

        if self.cache_output:
            out = self._encode_with_cache(flattened_utterances)
        else:
            out = self._encode_no_cache(flattened_utterances)

        if not self.freeze:
            out = self.batch_norm(out)

        # Reshape the batch of utterances into a list of utterances
        out = out.reshape(len(utterances), -1, self.model.config.hidden_size)

        return out

    @property
    def output_dim(self):
        return self.model.config.hidden_size


class ClassificationTaskModel(pl.LightningModule):
    """
    Class for the metric. It contains the metrics for the emotion and the trigger tasks.
    """

    def __init__(self, clf_input_size: int, clf_hidden_size: int = 128,
                 emotion_output_dim=7, trigger_output_dim=2, padding_value_emotion=7, padding_value_trigger=2, lr=1e-3,
                 class_weights_emotion: torch.Tensor | None = None, class_weights_trigger: torch.Tensor | None = None,
                 hidden_layers=1, dropout=0.2, alpha=None):
        super().__init__()

        self.emotion_output_dim = emotion_output_dim
        self.trigger_output_dim = trigger_output_dim
        self.padding_value_emotion = padding_value_emotion
        self.padding_value_trigger = padding_value_trigger
        self.clf_input_size = clf_input_size
        self.clf_hidden_size = clf_hidden_size
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        if alpha is not None:
            self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        else:
            self.alpha = None

        self.class_weights_emotion = class_weights_emotion.to(device) if class_weights_emotion is not None else None
        self.class_weights_trigger = class_weights_trigger.to(device) if class_weights_trigger is not None else None

        self.f1_cumulative_emotion = {}
        self.f1_cumulative_trigger = {}
        self.f1_dialogues_emotion = {}
        self.f1_dialogues_trigger = {}
        self.f1_dialogues_trigger_multi = {}
        self.f1_cumulative_trigger_multi = {}
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
            self.f1_cumulative_trigger_multi[stage] = F1ScoreCumulative(num_classes=self.trigger_output_dim,
                                                                        padding_value=self.padding_value_trigger,
                                                                        binary=False).to(device)
            self.f1_dialogues_trigger_multi[stage] = F1ScoreDialogues(num_classes=self.trigger_output_dim,
                                                                      padding_value=self.padding_value_trigger,
                                                                      binary=False).to(device)

        self.emotion_clf = CLF(self.clf_input_size, self.clf_hidden_size, self.emotion_output_dim, self.hidden_layers,
                               self.dropout)
        self.trigger_clf = CLF(self.clf_input_size, self.clf_hidden_size, self.trigger_output_dim, self.hidden_layers,
                               self.dropout)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def metric_update(self, stage: str, y_hat_class_emotion, y_emotion, y_hat_class_trigger, y_trigger):
        self.f1_cumulative_emotion[stage].update(y_hat_class_emotion, y_emotion)
        self.f1_cumulative_trigger[stage].update(y_hat_class_trigger, y_trigger)
        self.f1_cumulative_trigger_multi[stage].update(y_hat_class_trigger, y_trigger)

        self.f1_dialogues_emotion[stage].update(y_hat_class_emotion, y_emotion)
        self.f1_dialogues_trigger[stage].update(y_hat_class_trigger, y_trigger)
        self.f1_dialogues_trigger_multi[stage].update(y_hat_class_trigger, y_trigger)

    def on_epoch_type_end(self, type):
        self.log_dict({
            f'f1_{type}_cumulative_emotion': self.f1_cumulative_emotion[type].compute(),
            f'f1_{type}_cumulative_trigger': self.f1_cumulative_trigger[type].compute(),
            f'f1_{type}_cumulative_trigger_multi': self.f1_cumulative_trigger_multi[type].compute(),
            f'f1_{type}_dialogues_emotion': self.f1_dialogues_emotion[type].compute(),
            f'f1_{type}_dialogues_trigger': self.f1_dialogues_trigger[type].compute(),
            f'f1_{type}_dialogues_trigger_multi': self.f1_dialogues_trigger_multi[type].compute(),
        }, prog_bar=True, on_epoch=True, on_step=False)
        # Reset metrics
        self.f1_cumulative_emotion[type].reset()
        self.f1_cumulative_trigger[type].reset()
        self.f1_cumulative_trigger_multi[type].reset()

        self.f1_dialogues_emotion[type].reset()
        self.f1_dialogues_trigger[type].reset()
        self.f1_dialogues_trigger_multi[type].reset()

    def on_train_epoch_end(self):
        self.on_epoch_type_end('train')

    def on_validation_epoch_end(self):
        self.on_epoch_type_end('val')

    def on_test_epoch_end(self):
        self.on_epoch_type_end('test')

    def predict(self, batch):
        emotion_logits, trigger_logits = self(batch)
        emotion_logits, trigger_logits = self._transform_logits(emotion_logits, trigger_logits)
        #predictions = self._predict_from_logits(emotion_logits, trigger_logits)

        return {
            'emotions': emotion_logits,
            'triggers': trigger_logits
        }

    def _transform_logits(self, emotion_logits, trigger_logits):
        trigger_logits = trigger_logits.squeeze(-1)
        trigger_logits = torch.movedim(trigger_logits, 1, 2)
        emotion_logits = torch.movedim(emotion_logits, 1, 2)

        return emotion_logits, trigger_logits

    def _predict_from_logits(self, emotion_logits, trigger_logits):
        emotion_logits = torch.nn.Softmax(dim=1)(emotion_logits)
        trigger_logits = torch.nn.Softmax(dim=1)(trigger_logits)

        emotion_pred = torch.argmax(emotion_logits, dim=1)
        trigger_pred = torch.argmax(trigger_logits, dim=1)

        return emotion_pred, trigger_pred

    def type_step(self, batch, batch_idx, type):
        emotion_logits, trigger_logits = self(batch)
        emotion_logits, trigger_logits = self._transform_logits(emotion_logits, trigger_logits)

        y_emotion = batch['emotions'].to(device)
        y_trigger = batch['triggers'].to(device)

        # Compute metrics
        y_emotion_pred, y_trigger_pred = self._predict_from_logits(emotion_logits, trigger_logits)
        self.metric_update(type, y_emotion_pred, y_emotion, y_trigger_pred, y_trigger)

        # Compute loss
        emotion_loss_obj = torch.nn.CrossEntropyLoss(ignore_index=self.padding_value_emotion,
                                                     weight=self.class_weights_emotion)
        trigger_loss_obj = torch.nn.CrossEntropyLoss(ignore_index=self.padding_value_trigger,
                                                     weight=self.class_weights_trigger)

        emotion_loss = emotion_loss_obj(emotion_logits, y_emotion)
        trigger_loss = trigger_loss_obj(trigger_logits, y_trigger)

        loss = emotion_loss + trigger_loss

        # Weight the losses
        if self.alpha is not None:
            loss = (1 - self.alpha) * emotion_loss + self.alpha * trigger_loss

        batch_size = emotion_logits.size(0)
        self.log(f'{type}_trigger_loss', trigger_loss, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=batch_size)
        self.log(f'{type}_emotion_loss', emotion_loss, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=batch_size)
        self.log(f'{type}_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.type_step(batch, batch_idx, 'test')

    def pred(self, batch, batch_idx, dataloader_idx=None):
        emotion_logits, trigger_logits = self(batch)
        emotion_logits, trigger_logits = self._transform_logits(emotion_logits, trigger_logits)
        predictions = self._predict_from_logits(emotion_logits, trigger_logits)

        return {
            'emotions': predictions[0],
            'triggers': predictions[1]
        }
