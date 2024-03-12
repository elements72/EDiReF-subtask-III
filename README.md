# EDiReF-subtask-III a wide model comparison

This project was developed for the Natural Language Processing course in the Master's Degree in Artificial Intelligence at the University of Bologna.

### Abstract
This study explores emotion recognition intricacies in dialogues, addressing the SemEval2024 challenge with a dual focus on ERC and EFR. We center our experiments on BERT-based architectures. Starting with a simple baseline with no context we incorporated layers for capturing the global context, like an LSTM or an attention mechanism. This addition proves a relevant improvement over the EFR, but not on the ERC. To have a wider comparison we have also adapted EmoBERTa, well-known on the ERC task, to take into account the emotionflips problem. From our findings, the attention mechanism ensures the best performance on the EFR while all the EmoBERTa architectures are significantly better on the emotion recognition task.

### Description
For a detailed explanation of the project, please refer to [the report](https://github.com/elements72/EDiReF-subtask-III/blob/main/report.pdf).

### Usage
To utilize this project, follow these steps:

1. **Download the Repository:**
   Clone or download the repository to your local machine using the following command:

        git clone https://github.com/elements72/EDiReF-subtask-III.git

2. **Run the Notebook:**
   Execute the Jupyter Notebook named `main.ipynb` by running:

        jupyter notebook main.ipynb

This will open the notebook in your default web browser, allowing you to view and interact with the code.


## Contributing & Contact
For any contributions, questions, suggestions, or issues related to these assignments, please feel free to contact the authors:

- [Antonio Lopez](https://github.com/elements72)
- [Alessandra Blasioli](https://github.com/alessandrablasioli)
- [Matteo Vannucchi](https://github.com/MatteoVannucchi0)

These projects were developed as part of the coursework for the Natural Language Processing module at the University of Bologna. Your contributions are welcome and appreciated!
