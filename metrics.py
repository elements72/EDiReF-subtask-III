from torchmetrics import Metric
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score, BinaryF1Score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

class F1ScoreCumulative(Metric):
    def __init__(self, num_classes: int, padding_value: int = None, binary: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.mask = torch.zeros([num_classes], dtype=torch.bool).to(device)
        self.padding_value = padding_value if padding_value is not None else num_classes

        self.binary = binary

        if self.binary:
            # If the model is binary, we are interested in the F1 score of the positive class
            self.mask[[0]] = 1
        
        #self.mask[[padding_value]] = 0

        self.add_state("true_positive", default=torch.zeros([num_classes]), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.zeros([num_classes]), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.zeros([num_classes]), dist_reduce_fx="sum")

    def update(self, y_hat_class: torch.Tensor, y_class: torch.Tensor):

        confusion_matrix_metric = ConfusionMatrix(num_classes=self.num_classes, task="multiclass", ignore_index=self.padding_value).to(device)
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

        true_positive = torch.Tensor([confusion_matrix[i][i] for i in range(self.num_classes)]).to(device)
        false_negative = torch.Tensor([sum(confusion_matrix[i, :]) - true_positive[i] for i in range(self.num_classes)]).to(device)
        false_positive = torch.Tensor([sum(confusion_matrix[:, i]) - true_positive[i] for i in range(self.num_classes)]).to(device)

        self.true_positive += true_positive
        self.false_negative += false_negative
        self.false_positive += false_positive

    def compute(self):
        f1 = self.compute_category()
        # Create a mask that is False for NaNs
        mask = torch.isnan(f1) | self.mask

        # Invert the mask: True for valid entries, False for NaNs
        valid_data = f1[~mask]  

        # Compute the mean of the non-NaN values
        mean_value = torch.mean(valid_data)
        return mean_value

    def compute_category(self):
        precision = self.true_positive / (self.true_positive + self.false_positive)
        recall = self.true_positive / (self.true_positive + self.false_negative)

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1
    
    
    
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
            self.f1_score = MulticlassF1Score(num_classes=self.num_classes, ignore_index=self.padding_value, multidim_average='samplewise')

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_hat_class: torch.Tensor, y_class: torch.Tensor):
        if self.binary:
            # [batch_size, 2, seq_len] -> [batch_size, seq_length]
            y_hat_class = torch.argmax(y_hat_class, dim=-1)
        f1_score = self.f1_score(y_hat_class, y_class)
        self.sum += f1_score.sum()  
        self.n += f1_score.numel()

    def compute(self):
        return self.sum / self.n
