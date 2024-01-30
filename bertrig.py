import torch

from common_models import BertEncoder, ClassificationTaskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTrig(ClassificationTaskModel):

    def __init__(self, emotion_output_dim=7, trigger_output_dim=2, freeze_context=False, freeze_sentence=True,
                 sentence_encoder_name='bert-base-uncased', context_encoder_name='tae898/emoberta-large' ,use_encoder_cache: bool = True, encoder_cache_size: int = 100_000,
                 **kwargs):
        
        context_encoder = BertEncoder(context_encoder_name, emotion_output_dim, trigger_output_dim, freeze_context,
                              cache_output=False, cache_size=encoder_cache_size)
        
        sentence_encoder = BertEncoder(sentence_encoder_name, emotion_output_dim, trigger_output_dim, freeze_sentence,
                              cache_output=use_encoder_cache, cache_size=encoder_cache_size)
        
        clf_input_dim = context_encoder.output_dim + sentence_encoder.output_dim

        super().__init__(emotion_output_dim=emotion_output_dim, trigger_output_dim=trigger_output_dim,
                         clf_input_size=clf_input_dim, **kwargs)

        self.context_encoder = context_encoder
        self.encoder = sentence_encoder
        self.save_hyperparameters()

    def on_save_checkpoint(self, checkpoint):
        self.encoder.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.encoder.on_load_checkpoint(checkpoint)

    def forward(self, x):
        encoded_context = self.context_encoder.encode(x['context'])
        encoded_utterances = self.encoder.encode(x['utterances'])


        # lstm_out shape is [batch_size, hidden_size]
        # encoded_utterances shape is [batch_size, seq_len, bert_output_dim]
        # i need to concat the two tensors along the seq_len dimension
        # so that the final shape is [batch_size, seq_len, bert_output_dim + hidden_size]

        clf_input = torch.cat([encoded_utterances, encoded_context], dim=-1)

        # Run it through the classifier
        emotion_logits = self.emotion_clf(clf_input)
        trigger_logits = self.trigger_clf(clf_input)
        return emotion_logits.to(device), trigger_logits.to(device)
