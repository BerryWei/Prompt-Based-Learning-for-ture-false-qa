# Prompt-based learning for True-False-Question-Aanswering

```
┌── data
│   ├── true-false-qa-RAW
│   ├──  true-false-qa-TRAIN
│   └── true-false-qa-TEST
├── utlis
│   └── prompt.py
├── environment.yml
├── Makefile
├── RAW_data_preprocessing
├── train.sh
├── trainQA.py
└── job_v100_32g.sh
```

# In this repo, we use `hfl/chinese-roberta-wwm-ext` for the pretrained language model to do the binary classification task.

## 1. Setup the environment
```bash=
make
```
##  2. The training data is prepared in the `./data` folder.
```bash=
python RAW_data_preprocessing.py
```
##  3. Training prompt model
```bash=
bash train.sh
```



