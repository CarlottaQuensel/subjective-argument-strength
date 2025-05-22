################
# IMPORTS
################

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import os
import argparse
import random
import json

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding

import torch.optim as optim

from transformers import logging
logging.set_verbosity_error()

random.seed(42)


################
# MODEL CLASS
################
# wrapper class for any transformer model to use the forward method
class RoBERTa(nn.Module):
    """RoBERTa model: returns the prediction and the cross-entropy loss. Is loaded from 'model path'"""

    def __init__(self, encoder):
        super(RoBERTa, self).__init__()

        self.encoder = encoder

    def forward(self, input_ids, attention_mask, labels):
        loss, text_fea = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels)[:2]

        return loss, text_fea


################
# WEIGHTS SAVING
################
def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    """Save the model and that state_dict"""
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


################
# TRAINING
################
def train(model, optimizer, train_loader, valid_loader, num_epochs, destination_folder,
          best_valid_loss=float("Inf")):
    eval_every = len(train_loader) // 2
    running_loss = 0.0
    valid_running_loss = 0.0
    valid_running_f1 = 0.0
    global_step = 0
    best_valid_f1 = 0.0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for item in train_loader:
            input_ids = item['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = item['attention_mask']
            attention_mask = attention_mask.to(device)
            labels = item['labels'].type(torch.LongTensor)
            labels = labels.to(device)
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask, labels=labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for val_item in valid_loader:
                        val_input_ids = val_item['input_ids']
                        val_input_ids = val_input_ids.to(device)
                        val_attention_mask = val_item['attention_mask']
                        val_attention_mask = val_attention_mask.to(device)
                        val_labels = val_item['labels'].type(torch.LongTensor)
                        val_labels = val_labels.to(device)
                        output = model(
                            input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                        loss, out = output
                        y_pred = torch.argmax(out, 1).tolist()
                        y_true = val_labels.tolist()
                        valid_running_f1 += f1_score(y_true=y_true,
                                                     y_pred=y_pred, average="macro")
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_valid_f1 = valid_running_f1 / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                valid_running_f1 = 0.0
                model.train()

                # print progress
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{global_step}/{num_epochs * len(train_loader)}], Train Loss: {average_train_loss:.4f}, Valid Loss: {average_valid_loss:.4f}, Valid F1: {average_valid_f1:.4f}')
                #   .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                #           average_train_loss, average_valid_loss, average_valid_f1))

                # checkpoint
                if best_valid_f1 < average_valid_f1:
                    # best_valid_loss = average_valid_loss
                    best_valid_f1 = average_valid_f1
                    save_checkpoint(destination_folder +
                                    '/model.pt', model, best_valid_loss)
                    save_metrics(destination_folder + '/metrics.pt', train_loss_list, valid_loss_list,
                                 global_steps_list)

    save_metrics(destination_folder + '/metrics.pt',
                 train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


################
# EVALUATION
################
def evaluate(model, test_loader, result_folder):
    y_pred = []
    y_true = []
    y_scores = []
    predictions_path = result_folder + "predictions.csv"
    report_path = result_folder + "classification_report.csv"
    model.eval()
    with torch.no_grad():
        for item in test_loader:
            labels = item['labels'].type(torch.LongTensor)
            labels = labels.to(device)
            ids = item['input_ids'].type(torch.LongTensor)
            ids = ids.to(device)
            mask = item['attention_mask'].type(torch.LongTensor)
            mask = mask.to(device)
            output = model(ids, mask, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
            y_scores.extend(torch.softmax(output, 1).tolist())
    with open(predictions_path, "w") as f:
        f.write("gold label\tpredicted label\tprobability\n")
        for i in range(len(y_pred)):
            f.write(str(y_true[i]) + "\t" + str(y_pred[i]) +
                    "\t" + str(y_scores[i]) + "\n")
    f.close()
    report = classification_report(y_true, y_pred, labels=[
                                   0, 1], digits=2, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(report_path, sep="\t")
    return report


################
# PREDICTING
################
def predict(model, pred_loader):
    y_ids = []
    y_pred = []
    y_scores = []
    model.eval()
    with torch.no_grad():
        for item in pred_loader:
            arg_ids = item['text_id'].type(torch.LongTensor)
            arg_ids = arg_ids.to(device)
            ids = item['input_ids'].type(torch.LongTensor)
            ids = ids.to(device)
            mask = item['attention_mask'].type(torch.LongTensor)
            mask = mask.to(device)
            labels = item['labels'].type(torch.LongTensor)
            labels = labels.to(device)
            output = model(ids, mask, labels)
            _, output = output

            y_ids.extend(arg_ids.tolist())
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_scores.extend(torch.softmax(output, 1).tolist())

    return y_ids, y_pred, y_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mixed or cmv_only
    parser.add_argument("variant", type=str,
                        help="Variant of the training data to use: 'mixed' for the full set, 'cmv_only' for the subset")
    parser.add_argument("num_splits", type=int,
                        help="Number of splits k for k-fold ensemble classification")
    parser.add_argument("epochs", type=int,
                        help="Number of epochs for which to train the RoBERTa models")
    args = parser.parse_args()

    # Make output and temp folders if they don't exist
    dest_folder = "results/storytelling/" + args.variant
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Prepare training loop and relabeling info for splits
    num_splits = args.num_splits
    unclear_predictions = int(num_splits/2)
    relabel = {}
    for e in range(num_splits+1):
        if e <= unclear_predictions:
            relabel[e] = 0
        else:
            relabel[e] = 1

    # GPU if available, otherwise CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # TOKENIZER: init the tokenizer that corresponds to the model
    model_path = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)
    print(f"loaded tokenizer from <== {model_path}")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=512, return_tensors='pt')

    ############
    # ARGUMENT DATA
    # Load argument data first for prediction in each split
    ############
    # Load data into a DataFrame for later merging with the predictions
    ibm_df = pd.read_csv("data/arguments/ibm-argq_aggregated.csv", sep="\t")
    cmv_df = pd.read_csv("data/arguments/CMV_Cornell_2016.csv", sep="\t")
    # Convert into Datasets
    ibm_data = ibm_df  # [["text_id", "text"]]
    ibm_data["label"] = pd.Series([0 for i in range(len(ibm_data))])
    ibm_data = Dataset.from_pandas(ibm_data)
    cmv_data = cmv_df  # [["text_id", "text"]]
    cmv_data["label"] = pd.Series([0 for i in range(len(cmv_data))])
    cmv_data = Dataset.from_pandas(cmv_data)
    # Tokenize and set the right format for model input
    argument_data = DatasetDict({"ibm": ibm_data, "cmv": cmv_data})
    argument_data = argument_data.map(tokenize, batched=True)
    argument_data.set_format(
        "torch", columns=["text_id", "input_ids", "attention_mask", "label"])
    ibm_loader = DataLoader(
        argument_data["ibm"], batch_size=16, collate_fn=data_collator)
    cmv_loader = DataLoader(
        argument_data["cmv"], batch_size=16, collate_fn=data_collator)
    print("Loaded argument data from <== data/arguments/ibm-argq_aggregated.csv\n" +
          30*" "+"data/arguments/CMV_Cornell_2016.csv")
    # After the data has the right type for prediction, set the index of the original DataFrame for later
    ibm_df.set_index(("text_id"), inplace=True)
    cmv_df.set_index(("text_id"), inplace=True)

    ##############
    # STORYTELLING MODEL
    # Repeat training and predicting process 10 times for different dataset splits
    ##############
    storytelling_data = pd.read_csv(
        f"data/storytelling/storytelling_{args.variant}.csv", sep="\t")
    print("Loaded training data from <==",
          f"data/storytelling/storytelling_{args.variant}.csv")

    for split in range(num_splits):
        print(15*"=", f"Split {split+1}/{num_splits}", 15*"=")
        # Randomize selection of training and text/validation data differently according to each split
        train_df, val_test_df = train_test_split(
            storytelling_data, test_size=0.3, stratify=storytelling_data["label"], random_state=split)
        val_df, test_df = train_test_split(
            val_test_df, test_size=0.5, stratify=val_test_df["label"], random_state=42)

        # Tokenize training data and reformat into correct model input
        train_data = Dataset.from_pandas(train_df)
        val_data = Dataset.from_pandas(val_df)
        test_data = Dataset.from_pandas(test_df)
        data = DatasetDict(
            {'train': train_data, 'test': test_data, 'val': val_data})
        data = data.map(tokenize, batched=True)
        data.set_format("torch", columns=[
                        "input_ids", "attention_mask", "label"])

        # Iterators
        train_dataloader = DataLoader(
            data["train"], shuffle=True, batch_size=16, collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            data["val"], batch_size=16, collate_fn=data_collator
        )
        print("Initialized iterators")
        # init Roberta model
        encoder = RobertaForSequenceClassification.from_pretrained(model_path)
        model = RoBERTa(encoder).to(device)
        # init optimizer
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        print(
            30*"_" + f"\nInitialized model, begin training split {split+1}/{num_splits}")

        # Train the model
        train(model=model, optimizer=optimizer, train_loader=train_dataloader,
              valid_loader=eval_dataloader, destination_folder=dest_folder, num_epochs=args.epochs)

        ############
        # TESTING
        ############
        # Load the best model after trained for max epochs
        print(30*"_" + "\nNow testing...")
        best_model = RoBERTa(encoder).to(device)
        # Load best model from checkpoint (saved only when val f1 goes up in training)
        load_checkpoint(dest_folder + '/model.pt', best_model, device)
        test_dataloader = DataLoader(
            data["test"], batch_size=16, collate_fn=data_collator
        )
        # Evaluate the model on the test set
        eval_path = dest_folder + f"/{split}_"
        report = evaluate(best_model, test_dataloader, eval_path)

        print("Results...\n")
        print(
            f"No storytelling:\nPrecision: {report['0']['precision']:.2f}\tRecall: {report['0']['recall']:.2f}\tF1: {report['0']['f1-score']:.2f}")
        print(
            f"Storytelling:\nPrecision: {report['1']['precision']:.2f}\tRecall: {report['1']['recall']:.2f}\tF1: {report['1']['f1-score']:.2f}")

        ############
        # ARGUMENT DATA
        # Annotating argument data with storytelling labels with best_model
        ############
        print(30*"_" + "\nNow predicting on argument data...")
        # IBM ######
        y_ids, y_pred, y_scores = predict(best_model, ibm_loader)
        tmp = {"text_id": y_ids, f"prediction_{split}": y_pred,
               f"score_{split}": y_scores}
        tmp = pd.DataFrame(tmp)
        tmp.set_index(("text_id"), inplace=True)
        if split == 0:
            ibm_splits = tmp
        else:
            ibm_splits[f"prediction_{split}"] = tmp[f"prediction_{split}"]
            ibm_splits[f"score_{split}"] = tmp[f"score_{split}"]
        # CMV ######
        y_ids, y_pred, y_scores = predict(best_model, cmv_loader)
        tmp = {"text_id": y_ids, f"prediction_{split}": y_pred,
               f"score_{split}": y_scores}
        tmp = pd.DataFrame(tmp)
        tmp.set_index(("text_id"), inplace=True)
        if split == 0:
            cmv_splits = tmp
        else:
            cmv_splits[f"prediction_{split}"] = tmp[f"prediction_{split}"]
            cmv_splits[f"score_{split}"] = tmp[f"score_{split}"]

    #############
    # Combining split results
    #############
    # After all splits are trained and predicted on, save the detailed predictions with probability scores per split
    ibm_splits.to_csv(
        dest_folder+"/ibm_storytelling_predictions.csv", sep="\t")
    cmv_splits.to_csv(
        dest_folder+"/cmv_storytelling_predictions.csv", sep="\t")
    # and aggregate the prediction results into one majority class by first summing over all predictions,
    ibm_splits.drop(
        columns=(f"score_{s}" for s in range(num_splits)), inplace=True)
    cmv_splits.drop(
        columns=(f"score_{s}" for s in range(num_splits)), inplace=True)
    # saving uncertain predictions (5 positive, 5 negative splits) for manual evaluation,
    ibm_results = ibm_splits.sum(axis=1)
    cmv_results = cmv_splits.sum(axis=1)
    ibm_index = ibm_results.index
    cmv_index = cmv_results.index
    rand_ids = {"ibm": ibm_index[ibm_results == unclear_predictions].tolist(
    ), "cmv": cmv_index[cmv_results == unclear_predictions].tolist()}
    with open(dest_folder+"/storytelling_unclear_predictions.json", "w") as f:
        json.dump(rand_ids, f, indent=4)
    # and then setting the label to 1 with at least half of the splits predicting 1
    ibm_results.replace(relabel, inplace=True)
    cmv_results.replace(relabel, inplace=True)
    ibm_df["storytelling"] = ibm_results
    cmv_df["storytelling"] = cmv_results
    ibm_df.drop(columns=("label"), inplace=True)
    cmv_df.drop(columns=("label"), inplace=True)
    ibm_df.to_csv(
        dest_folder+f"/ibm_with_storytelling_{args.variant}.csv", sep="\t")
    cmv_df.to_csv(
        dest_folder+f"/cmv_with_storytelling_{args.variant}.csv", sep="\t")
