# CS5562-HW2-report

This is the second homework assignment for CS5562 Trustworthy ML. Please clone this repo or download all the Python files and proceed with the instructions on the PDF.

Download the clean model file from [here](https://drive.google.com/file/d/1e4xLwkxq2VLMmfY8kCRBvgjw8o0hYF9y/view?usp=drive_link).

- Question 1
The poisoned dataset file is stored in `data/SST2_poisoned/train.tsv`

- Question 2

The backdoord model file is stored in `SST2_EP_model`.

- Question 3

The final poisoned test data file from the last iteration is stored in `data/SST2_poisoned/final_poisoned_test_last_itr.tsv`.

The output for Clean Model is:

==========Computing ASR and clean accuracy on test dataset==========
Trigger word: bb
Model: SST2_clean_model
Repetition:  0
Repetition:  1
Repetition:  2
        Clean Test Loss: 0.246 | Clean Test Acc: 92.99%
        Poison Test Loss: 4.554 | Poison Test Acc: 5.84%

The output for EP Model is:

==========Computing ASR and clean accuracy on test dataset==========
Trigger word: bb
Model: SST2_EP_model
Repetition:  0
Repetition:  1
Repetition:  2
        Clean Test Loss: 0.246 | Clean Test Acc: 92.99%
        Poison Test Loss: 0.001 | Poison Test Acc: 100.00%
