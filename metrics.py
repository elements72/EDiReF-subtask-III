from torchmetrics import Metric
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score
import torch


class F1ScoreCumulative(Metric):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes
        self.mask = torch.ones([num_classes], dtype=torch.bool)
        #self.mask[[padding_value]] = 0

        self.add_state("true_positive", default=torch.zeros([num_classes]), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.zeros([num_classes]), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.zeros([num_classes]), dist_reduce_fx="sum")

    def update(self, y_hat_class: torch.Tensor, y_class: torch.Tensor):
        confusion_matrix_metric = ConfusionMatrix(num_classes=self.num_classes, task="multiclass", ignore_index=-1)
        confusion_matrix = confusion_matrix_metric(y_hat_class, y_class)

        # #  Example of multiclass confusion matrix:
        # #  Confusion matrix, TP, FN and FP for class 0 
        # #   TRUE LABEL
        # #   0               TP     FN     FN     FN     FN       
        # #   1               FP       
        # #   2               FP               
        # #   3               FP                       
        # #   4               FP                                
        # # PREDICTED LABEL   0       1      2      3      4 

        true_positive = torch.Tensor([confusion_matrix[i][i] for i in range(self.num_classes)])
        false_negative = torch.Tensor([sum(confusion_matrix[i, :]) - true_positive[i] for i in range(self.num_classes)])
        false_positive = torch.Tensor([sum(confusion_matrix[:, i]) - true_positive[i] for i in range(self.num_classes)])

        self.true_positive += true_positive
        self.false_negative += false_negative
        self.false_positive += false_positive

    def compute(self):
        precision = self.true_positive / (self.true_positive + self.false_positive)
        recall = self.true_positive / (self.true_positive + self.false_negative)

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1
    
class F1ScoreDialogues(Metric):
    '''
    F1 score per dialogue.
    For each dialogue we compute the F1 score and we avergae over all dialogues.
    '''
    def __init__(self, num_classes: int, padding_value: int = -1):
        super().__init__()

        self.num_classes = num_classes
        self.f1_score = MulticlassF1Score(num_classes=self.num_classes, average='macro', ignore_index=padding_value, multidim_average='samplewise')

        self.add_state("sum", default=torch.float32, dist_reduce_fx="sum")
        self.add_state("n", default=torch.int, dist_reduce_fx="sum")

    def update(self, y_hat_class: torch.Tensor, y_class: torch.Tensor):
        f1_score = self.f1_score(y_hat_class, y_class)
        self.sum += f1_score.sum()
        self.n += f1_score.numel()

    def compute(self):
        return self.sum / self.n
