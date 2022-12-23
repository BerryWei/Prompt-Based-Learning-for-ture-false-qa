from argparse import ArgumentParser, Namespace
import torch
import numpy as np
import sys
from transformers import AdamW
import torch.optim as optim
from tqdm import trange
import logging
import os
from pathlib import Path
from utlis.prompt import get_examples
from typing import *
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
import matplotlib.pyplot as plt
import pickle


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Prompt Learning on chinese pretrained model!")
    parser.add_argument(
        "--train_data",
        type=Path,
        help="Directory to the training data.",
        default="./data/true-false-qa-TRAIN.json",
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        help="Directory to the testing data.",
        default="./data/true-false-qa-TEST.json",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--with_plotting",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    
    # model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=5)

    args = parser.parse_args()
    
    # Sanity checks
    if args.train_data is None or args.test_data is None:
        raise ValueError("Need --train_data AND --test_data")
    return args

def get_loggings(ckpt_dir):
    logger = logging.getLogger(name='TASK-Prompt-based Learning')
    logger.setLevel(level=logging.DEBUG)
    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # file handler
    file_handler = logging.FileHandler(os.path.join(ckpt_dir, "record.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main(args):
    args = parse_args()
    logger = get_loggings(args.ckpt_dir)
    #################################################
    ## Step 1. Define a task
    #################################################
    trainDataset = get_examples(args.train_data)
    validDataset = get_examples(args.test_data)
    classes = [ "negative", "positive" ]
    assert isinstance(trainDataset, List)
    assert isinstance(trainDataset[0], InputExample)
    assert isinstance(validDataset, List)
    assert isinstance(validDataset[0], InputExample)

    #################################################
    ## Step 2. Obtain a PLM
    #################################################
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", args.model_name_or_path)
    
    #################################################
    ## Step 3. Define a Template
    #################################################
    promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It was {"mask"}',
    tokenizer = tokenizer,
    )

    #################################################
    ## Step 4. Define a Verbalizer
    #################################################
    promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer = tokenizer,
    )

    #################################################
    ## Step 5. Construct a PromptModel
    #################################################

    promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
    )

    #################################################
    ## Step 6. Define a DataLoader
    #################################################
    train_loader = PromptDataLoader(
        dataset = trainDataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = args.batch_size,
        shuffle=True,
    )
    valid_loader = PromptDataLoader(
        dataset = validDataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = args.batch_size,
    )
    logger.info(f"train, valid dataset len: { len(trainDataset) }, { len(validDataset) }")
    logger.info(promptModel)
    #################################################
    ## Step 7. Training
    #################################################
    promptModel.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_eval_loss = np.inf
    # declare variable for plotting
    loss_train_list = []
    loss_eval_list = []
    acc_train_list = []
    acc_eval_list = []


    for epoch in epoch_pbar:
        promptModel.train()
        loss_train, acc_train, iter_train = 0, 0, 0
        for step, items in enumerate(train_loader):
            items = items.to(args.device)
            labels = items['label']
            output = promptModel(items)

            # calculate loss and update parameters
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss, accuracy
            iter_train += 1
            loss_train += loss.item()
            pred = output.max(1, keepdim=True)[1]
            acc_train += pred.eq(labels.view_as(pred)).sum().item()

        loss_train /= len(trainDataset)
        acc_train  /= len(trainDataset)
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)


        # Evaluation loop - calculate accuracy and save model weights
        promptModel.eval()
        with torch.no_grad():
            loss_eval, acc_eval, iter_eval = 0, 0, 0
            for step, inputs in enumerate(valid_loader):
                items = items.to(args.device)
                labels = items['label']
                output = promptModel(items)

                # calculate loss and update parameters
                loss = criterion(output, labels)

                # accumulate loss, accuracy
                iter_eval += 1
                loss_eval += loss.item()
                pred = output.max(1, keepdim=True)[1]
                acc_eval += pred.eq(labels.view_as(pred)).sum().item()

            loss_eval /= len(validDataset)
            acc_eval  /= len(validDataset)
            loss_eval_list.append(loss_eval)
            acc_eval_list.append(acc_eval)

        logger.info(f"epoch: {epoch}, train_acc: {acc_train:.5f}, eval_acc: {acc_eval:.5f}, train_loss: {loss_train:.5f}, eval_loss: {loss_eval:.5f}")
        sys.stdout.flush()
        scheduler.step(loss_eval)

        # save model
        if loss_eval < best_eval_loss:
            best_eval_loss = loss_eval
            logger.info(f"Trained model saved, eval loss: {best_eval_loss:.4f}")
            best_model_path = args.ckpt_dir / 'model.pt'
            torch.save(promptModel.state_dict(), best_model_path)

    # save plotting variables
    VARS_learning_curve = {
        'epochs': [i for i in range(args.num_epoch)],
        'loss_train': loss_train_list,
        'acc_train': acc_train_list,
        'loss_eval': loss_eval_list,
        'acc_eval': acc_eval_list,
    }
    savepath = os.path.join(args.ckpt_dir, "VARS_learning_curve.pkl")
    with open(savepath, 'wb') as file:
        pickle.dump(VARS_learning_curve, file)

    if args.with_plotting:
        plt.figure(dpi = 300)
        plt.title('loss curve')
        plt.plot(VARS_learning_curve['epochs'], VARS_learning_curve['loss_train'], label='loss_train')
        plt.plot(VARS_learning_curve['epochs'], VARS_learning_curve['loss_eval'],  label='loss_eval')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        savepath = os.path.join(args.ckpt_dir, "loss.png")
        plt.savefig(savepath)

        plt.figure(dpi = 300)
        plt.title('acc curve')
        plt.plot(VARS_learning_curve['epochs'], VARS_learning_curve['acc_train'], label='acc_train')
        plt.plot(VARS_learning_curve['epochs'], VARS_learning_curve['acc_eval'], label='acc_eval')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.grid()
        plt.legend()
        savepath = os.path.join(args.ckpt_dir, "acc.png")
        plt.savefig(savepath)
    
    






if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)