import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassF1Score, BinaryF1Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


class F1ScoreCumulative(Metric):
    def __init__(self, num_classes: int, padding_value: int = None, binary: bool = False):
        super().__init__()

        self.padding_value = padding_value if padding_value else num_classes
        self.num_classes = num_classes
        self.binary = binary

        if not self.binary:
            self.f1_score = MulticlassF1Score(num_classes=self.num_classes, ignore_index=self.padding_value)
        else:
            self.f1_score = BinaryF1Score(ignore_index=self.padding_value)

        print('F1ScoreCumulative: binary = ', self.binary)

    def update(self, y_hat_class: torch.Tensor, y_class: torch.Tensor):
        self.f1_score.update(y_hat_class, y_class)

    def compute(self):
        return self.f1_score.compute()

    def reset(self):
        self.f1_score.reset()


class F1ScoreDialogues(Metric):
    '''
    F1 score per dialogue.
    For each dialogue we compute the F1 score and we avergae over all dialogues.
    '''

    def __init__(self, num_classes: int, padding_value: int = None, binary: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.padding_value = padding_value if padding_value else num_classes
        self.binary = binary
        if binary:
            self.f1_score = BinaryF1Score(ignore_index=self.padding_value, multidim_average='samplewise')
        else:
            self.f1_score = MulticlassF1Score(num_classes=self.num_classes, ignore_index=self.padding_value,
                                              multidim_average='samplewise')

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_hat_class: torch.Tensor, y_class: torch.Tensor):
        # if self.binary:
        #     # Apply softmax to get the probability of the positive class
        #     y_hat_class = torch.nn.functional.softmax(y_hat_class, dim=1)
        #     y_hat_class = torch.argmax(y_hat_class, dim=1)
        f1_score = self.f1_score(y_hat_class, y_class)
        self.sum += f1_score.sum()
        self.n += f1_score.numel()

    def compute(self):
        return self.sum / self.n
