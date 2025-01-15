import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import logging
import argparse
import os
import re
import random
import shutil
import json

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
from transformers.data.metrics import acc_and_f1
from tqdm import tqdm, trange
from collections import defaultdict
from typing import List
from transformers import BertTokenizer, AutoModelForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter

tqdm.pandas()

def clean_text(text):
  # Remove b' at the beginning
  text = re.sub("^b'", "", text)
  # Remove @RT
  text = re.sub("RT @[\S]+", "", text)
  # Remove mention @~
  text = re.sub("@[\S]+", "", text)
  # Remove &#~
  text = re.sub("&#[\S]+", "", text)
  # remove URL
  url = re.compile(r'https?://\S+|www\.\S+')
  text = url.sub(r'',text)
  # remove HTML
  html=re.compile(r'<.*?>')
  text = html.sub(r'',text)
  # remove emoji
  emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              "]+", flags=re.UNICODE)
  text = emoji_pattern.sub(r'', text)
  # remove punctuation except 
  text = re.sub(r"[^\w\d'\s]+",'',text)
  # replace \n to ' '
  text = re.sub(r"\n+",' ',text).strip()
  # replace \t to ' '
  text = re.sub(r"\t+",' ',text).strip()
  # remove double spaces
  text = re.sub(r"\s+",' ',text).strip()
  return text


def preprocess(data):
    df = data.copy()
    
    # 데이터 소스 식별 및 적절한 컬럼 이름 설정
    if 'tweet' in df.columns and 'class' in df.columns:
        df.rename(columns={'tweet': 'post', 'class': 'offensiveYN'}, inplace=True)
        df["offensiveYN"] = df["offensiveYN"].apply(lambda x: 1 if x == 1 else 0)
    elif 'comment' in df.columns and 'isHate' in df.columns:
        df.rename(columns={'comment': 'post', 'isHate': 'offensiveYN'}, inplace=True)
        # 'isHate' 컬럼에 대한 추가 변환은 필요에 따라 조정
    elif 'post' in df.columns and 'offensiveYN' in df.columns:
        df["offensiveYN"] = df["offensiveYN"].apply(lambda x: 1 if x >= 0.5 else 0)
    else:
        raise ValueError("Unsupported data format")
    
    # 모든 데이터에 공통적으로 적용되는 텍스트 전처리
    df["post"] = df["post"].apply(lambda x: clean_text(x))
    
    return df[["offensiveYN", "post"]]



class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, index):
        input_ids = self.tokens["input_ids"][index]
        token_type_ids = self.tokens["token_type_ids"][index]
        attention_mask = self.tokens["attention_mask"][index]
        label = self.labels[index]

        return input_ids, token_type_ids, attention_mask, label, index
    
    def __len__(self):
        return len(self.labels)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate(args, model, tokenizer, prefix="", eval_split="dev"):
    # We do not really need a loop to handle MNLI double evaluation (matched, mis-matched).

    eval_output_dir = args.output_dir

    results = {}
    eval_data = pd.read_csv(args.eval_data_dir)
    eval_data = preprocess(eval_data)
    eval_tokenized = tokenizer(eval_data["post"].values.tolist(), 
                                padding=True, truncation=True, max_length=args.max_seq_length, return_tensors="pt")
    eval_dataset = CustomDataset(eval_tokenized, eval_data["offensiveYN"])
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running {prefix} evaluation on {eval_split} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    example_ids = []
    gold_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=10, ncols=100):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
            example_ids += batch[4].tolist()
            gold_labels += batch[3].tolist()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    probs = torch.nn.functional.softmax(torch.Tensor(preds), dim=-1)
    max_confidences = (torch.max(probs, dim=-1)[0]).tolist()
    preds = np.argmax(preds, axis=1)  # Max of logit is the same as max of probability.

    result = acc_and_f1(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(
        eval_output_dir, f"eval_metrics_{eval_split}_{prefix}.json")
    logger.info(f"***** {eval_split} results {prefix} *****")
    for key in sorted(result.keys()):
        logger.info(f"{eval_split} {prefix} {key} = {result[key]:.4f}")
    with open(output_eval_file, "a") as writer:
        writer.write(json.dumps(results) + "\n")

    # predictions
    all_predictions = []
    output_pred_file = os.path.join(
        eval_output_dir, f"predictions_{eval_split}_{prefix}.lst")
    with open(output_pred_file, "w") as writer:
        logger.info(f"***** Write {eval_split} predictions {prefix} *****")
        for ex_id, pred, gold, max_conf, prob in zip(
            example_ids, preds, gold_labels, max_confidences, probs.tolist()):
            record = {"guid": int(ex_id),
                      "label": int(pred),
                      "gold": int(gold),
                      "confidence": int(max_conf),
                      "probabilities": prob}
            all_predictions.append(record)
            writer.write(json.dumps(record) + "\n")

    return results, all_predictions


def save_model(args, model, tokenizer, epoch, best_epoch,  best_dev_performance, writer):
    results, _ = evaluate(args, model, tokenizer, prefix="in_training")
    # TODO(SS): change hard coding `acc` as the desired metric, might not work for all tasks.
    desired_metric = "acc"
    dev_performance = results.get(desired_metric)
    writer.add_scalar("ACC/DEV", dev_performance, epoch)
    if dev_performance > best_dev_performance:
        best_epoch = epoch
        best_dev_performance = dev_performance

        # Save model checkpoint
        # Take care of distributed/parallel training
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
 
        logger.info(f"*** Found BEST model, and saved checkpoint. "
            f"BEST dev performance : {dev_performance:.4f} ***")
    return best_dev_performance, best_epoch

def log_training_dynamics(output_dir: os.path,
                          epoch: int,
                          train_ids: List[int],
                          train_logits: List[List[float]],
                          train_golds: List[int]):
  """
  Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
  """
  td_df = pd.DataFrame({"guid": train_ids,
                        f"logits_epoch_{epoch}": train_logits,
                        "gold": train_golds})

  logging_dir = os.path.join(output_dir, f"training_dynamics")
  # Create directory for logging training dynamics, if it doesn't already exist.
  if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
  epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
  td_df.to_json(epoch_file_name, lines=True, orient="records")
  logger.info(f"Training Dynamics logged to {epoch_file_name}")


def train(args, train_dataset, model, tokenizer, train_batch_size):
  """ Train the model """
  if args.local_rank in [-1, 0]:
      tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

  train_sampler = RandomSampler(
      train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(
      train_dataset, sampler=train_sampler, batch_size=train_batch_size)
  if args.max_steps > 0:
      t_total = args.max_steps
      args.num_train_epochs = args.max_steps // (
          len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
      t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        },
      {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0
        },
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  # Train!
  # logger.info("***** Running training *****")
  # logger.info("  Num examples = %d", len(train_dataset))
  # logger.info("  Num Epochs = %d", num_train_epochs)
  # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  # logger.info(
  #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
  #     train_batch_size
  #     * args.gradient_accumulation_steps
  #     * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
  # )
  # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  # logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  epochs_trained = 0
  steps_trained_in_this_epoch = 0

  tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(epochs_trained,
                          int(args.num_train_epochs),
                          desc="Epoch",
                          disable=args.local_rank not in [-1, 0],
                          mininterval=10,
                          ncols=100)
  set_seed(args)  # Added here for reproductibility
  best_dev_performance = 0
  best_epoch = epochs_trained

  train_acc = 0.0
  for epoch, _ in enumerate(train_iterator):
      epoch_iterator = tqdm(train_dataloader,
                            desc="Iteration",
                            disable=args.local_rank not in [-1, 0],
                            mininterval=10,
                            ncols=100)

      train_iterator.set_description(f"train_epoch: {epoch} train_acc: {train_acc:.4f}")
      train_ids = None
      train_golds = None
      train_logits = None
      train_losses = None
      for step, batch in enumerate(epoch_iterator):
          # Skip past any already trained steps if resuming training
          if steps_trained_in_this_epoch > 0:
              steps_trained_in_this_epoch -= 1
              continue

          model.train()
          batch = tuple(t.to(args.device) for t in batch)
          inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids": batch[2], "labels": batch[3]}
          outputs = model(**inputs)
          loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

          if train_logits is None:  # Keep track of training dynamics.
              train_ids = batch[4].detach().cpu().numpy()
              train_logits = outputs[1].detach().cpu().numpy()
              train_golds = inputs["labels"].detach().cpu().numpy()
              train_losses = loss.detach().cpu().numpy()
          else:
              train_ids = np.append(train_ids, batch[4].detach().cpu().numpy())
              train_logits = np.append(train_logits, outputs[1].detach().cpu().numpy(), axis=0)
              train_golds = np.append(train_golds, inputs["labels"].detach().cpu().numpy())
              train_losses = np.append(train_losses, loss.detach().cpu().numpy())

          if args.n_gpu > 1:
              loss = loss.mean()  # mean() to average on multi-gpu parallel training
          if args.gradient_accumulation_steps > 1:
              loss = loss / args.gradient_accumulation_steps

          loss.backward()

          tr_loss += loss.item()
          if (step + 1) % args.gradient_accumulation_steps == 0:
              torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

              optimizer.step()
              scheduler.step()  # Update learning rate schedule
              model.zero_grad()
              global_step += 1

              if (
                  args.local_rank in [-1, 0] and
                  args.logging_steps > 0 and
                  global_step % args.logging_steps == 0
              ):
                  epoch_log = {}
                  # Only evaluate when single GPU otherwise metrics may not average well
                  if args.local_rank == -1 and args.evaluate_during_training_epoch:
                      logger.info(f"From within the epoch at step {step}")
                      results, _ = evaluate(args, model, tokenizer)
                      for key, value in results.items():
                          eval_key = "eval_{}".format(key)
                          epoch_log[eval_key] = value

                  epoch_log["learning_rate"] = scheduler.get_lr()[0]
                  epoch_log["loss"] = (tr_loss - logging_loss) / args.logging_steps
                  logging_loss = tr_loss

                  for key, value in epoch_log.items():
                      tb_writer.add_scalar(key, value, global_step)
                  logger.info(json.dumps({**epoch_log, **{"step": global_step}}))

              if (
                  args.local_rank in [-1, 0] and
                  args.save_steps > 0 and
                  global_step % args.save_steps == 0
              ):
                  # Save model checkpoint
                  output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                  if not os.path.exists(output_dir):
                      os.makedirs(output_dir)
                  model_to_save = (
                      model.module if hasattr(model, "module") else model
                  )  # Take care of distributed/parallel training
                  model_to_save.save_pretrained(output_dir)
                  tokenizer.save_pretrained(output_dir)

                  logger.info("Saving model checkpoint to %s", output_dir)

                  torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                  torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                  logger.info("Saving optimizer and scheduler states to %s", output_dir)

          # epoch_iterator.set_description(f"lr = {scheduler.get_lr()[0]:.8f}, "
          #                                 f"loss = {(tr_loss-epoch_loss)/(step+1):.4f}")
          if args.max_steps > 0 and global_step > args.max_steps:
              epoch_iterator.close()
              break

      #### Post epoch eval ####
      # Only evaluate when single GPU otherwise metrics may not average well
      if args.local_rank == -1 and args.evaluate_during_training:
          best_dev_performance, best_epoch = save_model(args,
              model, tokenizer, epoch, best_epoch, best_dev_performance, tb_writer)

      # Keep track of training dynamics.
      log_training_dynamics(output_dir=args.output_dir,
                            epoch=epoch,
                            train_ids=list(train_ids),
                            train_logits=list(train_logits),
                            train_golds=list(train_golds))
      train_result = acc_and_f1(np.argmax(train_logits, axis=1), train_golds)
      train_acc = train_result["acc"]

      epoch_log = {"epoch": epoch,
                    "train_acc": train_acc,
                    "best_dev_performance": best_dev_performance,
                    "avg_batch_loss": (tr_loss - epoch_loss) / args.per_gpu_train_batch_size,
                    "learning_rate": scheduler.get_lr()[0],}
      epoch_loss = tr_loss

      logger.info(f"  End of epoch : {epoch}")
      with open(os.path.join(args.output_dir, f"eval_metrics_train.json"), "a") as toutfile:
          toutfile.write(json.dumps(epoch_log) + "\n")
      for key, value in epoch_log.items():
          tb_writer.add_scalar(key, value, global_step)
          logger.info(f"  {key}: {value:.6f}")

      if args.max_steps > 0 and global_step > args.max_steps:
          train_iterator.close()
          break
      elif args.evaluate_during_training and epoch - best_epoch >= args.patience:
          logger.info(f"Ran out of patience. Best epoch was {best_epoch}. "
              f"Stopping training at epoch {epoch} out of {args.num_train_epochs} epochs.")
          train_iterator.close()
          break

  if args.local_rank in [-1, 0]:
      tb_writer.close()

  return global_step, tr_loss / global_step

class TrainArgs():
  def __init__(self, args):

    ### Required parameters

    # Input data
    # self.train : str = configs.get("train", None)
    # self.dev : str = configs.get("dev", None)
    # self.test: str = configs.get("test", None)
    
    # Pretrained model name
    self.model_name = args.model_name

    # Data directory for training
    self.train_data_dir = args.train_data_dir
    self.eval_data_dir = args.eval_data_dir
    # self.test_data_dir = args.test_data_dir
    self.sbic_data_dir = args.sbic_data_dir
    self.olid_data_dir = args.olid_data_dir
    self.dghd_data_dir = args.dghd_data_dir
    self.ethos_data_dir = args.ethos_data_dir
    self.toxigen_data_dir = args.toxigen_data_dir

    # Random seed for initialization.
    self.seed = args.seed

    # The output directory where the model predictions and checkpoints will be written.
    self.output_dir = args.output_dir

    self.lr = args.lr

    self.num_train_epochs = args.num_train_epochs

    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Whether to run training.
    # self.do_train : bool = configs.get("do_train", False)

    # # Whether to run eval on the dev set.
    # self.do_eval : bool = configs.get("do_eval", False)

    # # Whether to run eval on the dev set.
    # self.do_test : bool = configs.get("do_test", False)

    ### Other parameters

    # Where to store the feature cache for the model.
    self.features_cache_dir = "./feature_cache{}/".format(args.seed)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated,
    # sequences shorter will be padded.
    self.max_seq_length = 128

    # Run evaluation during training after each epoch.
    self.evaluate_during_training = True

    # Run evaluation during training at each logging step.
    self.evaluate_during_training_epoch =  False

    # Set this flag if you are using an uncased model.
    self.do_lower_case = True

    # Batch size per GPU/CPU for training.
    self.per_gpu_train_batch_size = args.batch_size

    # Batch size per GPU/CPU for evaluation.
    self.per_gpu_eval_batch_size = args.batch_size

    # Number of updates steps to accumulate before
    # performing a backward/update pass.
    self.gradient_accumulation_steps = 1

    # The initial learning rate for Adam.

    # Weight decay if we apply some.
    self.weight_decay = 0.0

    # Epsilon for Adam optimizer.
    self.adam_epsilon = 1e-8

    # Max gradient norm.
    self.max_grad_norm = 1.0


    # If > 0 : set total number of training steps to perform.
    # Override num_train_epochs.
    self.max_steps = -1

    # Linear warmup over warmup_steps.
    self.warmup_steps = 0

    # Log every X updates steps.
    self.logging_steps = 1000

    # If dev performance does not improve in X updates, end training.
    if args.do_early_stopping:
        self.patience = args.patience
        logger.info("Early stopping enabled")
    else:
        self.patience = args.num_train_epochs
        logger.info("Early stopping disabled")

    # Save checkpoint every X updates steps.
    self.save_steps = 0

    # Evaluate all checkpoints starting with the same prefix as
    # model_name ending and ending with step number
    self.eval_all_checkpoints = False

    # Avoid using CUDA when available
    self.no_cuda = False

    # Overwrite the content of the output directory
    self.overwrite_output_dir = False

    # Overwrite the cached training and evaluation sets
    self.overwrite_cache = False

    # Whether to use 16-bit (mixed) precision (through NVIDIA apex)
    # instead of 32-bit
    self.fp16 = False

    # For fp16 : Apex AMP optimization level selected in
    # ['O0', 'O1', 'O2', and 'O3'].
    # See details at https://nvidia.github.io/apex/amp.html"
    self.fp16_opt_level = "01"

    # For distributed training.
    self.local_rank = -1

    # For distant debugging.
    self.server_ip =  ""
    self.server_port = ""

    # number of gpu
    self.n_gpu = torch.cuda.device_count()


def run_train_dynamics(args):
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

    data = pd.read_csv(args.train_data_dir)
    data = preprocess(data)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels = 2).to(args.device)
    # Prepare train dataset
    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_tokenized = tokenizer(data["post"].values.tolist(), 
                                padding=True, truncation=True, max_length=args.max_seq_length, return_tensors="pt")
    train_dataset = CustomDataset(train_tokenized, data["offensiveYN"])
    train(args, train_dataset, model, tokenizer, train_batch_size)


    # 검증 데이터셋 평가
    args.eval_data_dir = args.eval_data_dir  # 검증 데이터셋 경로 지정
    logger.info("Evaluating on dev dataset...")
    eval_results, _ = evaluate(args, model, tokenizer, prefix="dev")
    logger.info(f"Dev Evaluation Results: {eval_results}")

    # 테스트 데이터셋 평가
    args.eval_data_dir = args.sbic_data_dir  # 테스트 데이터셋 경로로 변경
    logger.info("Evaluating on sbic dataset...")
    test_results, _ = evaluate(args, model, tokenizer, prefix="sbic")
    logger.info(f"SBIC Evaluation Results: {test_results}")


    # 테스트 데이터셋 평가
    args.eval_data_dir = args.olid_data_dir  # 테스트 데이터셋 경로로 변경
    logger.info("Evaluating on olid dataset...")
    olid_results, _ = evaluate(args, model, tokenizer, prefix="olid")
    logger.info(f"OLID Evaluation Results: {olid_results}")

    # 테스트 데이터셋 평가
    args.eval_data_dir = args.dghd_data_dir  # 테스트 데이터셋 경로로 변경
    logger.info("Evaluating on dyna_hate dataset...")
    dghd_results, _ = evaluate(args, model, tokenizer, prefix="dyna_hate")
    logger.info(f"DYNA HATE Evaluation Results: {dghd_results}")

    # 테스트 데이터셋 평가
    args.eval_data_dir = args.ethos_data_dir  # 테스트 데이터셋 경로로 변경
    logger.info("Evaluating on ethos dataset...")
    ethos_results, _ = evaluate(args, model, tokenizer, prefix="ethos")
    logger.info(f"Test Evaluation Results: {ethos_results}")
    
    args.eval_data_dir = args.toxigen_data_dir  # 테스트 데이터셋 경로로 변경
    logger.info("Evaluating on toxigen dataset...")
    toxigen_results, _ = evaluate(args, model, tokenizer, prefix="toxigen")
    logger.info(f"Test Evaluation Results: {toxigen_results}")


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="", type=int, default=42)
    parser.add_argument("--output_dir", help="model predictions and checkpoints will be written", type=str, default="./output")
    parser.add_argument("--lr", help="Learning rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--num_train_epochs", help="Number of epochs", type=int, default=6)
    parser.add_argument("--do_early_stopping", help="Do early_stopping", action="store_true")
    parser.add_argument("--patience", help="Patience until early stop", type=int, default=3)
    parser.add_argument("--model_name", help="Name of pretrained model", type=str, default="bert-base-uncased")
    parser.add_argument("--train_data_dir", help="Directory of data to train", type=str, required=True)
    parser.add_argument("--eval_data_dir", help="Directory of data to evaluate", type=str, required=True)
    # parser.add_argument("--test_data_dir", help="Directory of data to test", type=str, required=True) 
    parser.add_argument("--sbic_data_dir", help="Directory of sbic data", type=str, required=True)
    parser.add_argument("--olid_data_dir", help="Directory of olid data", type=str, required=True)
    parser.add_argument("--dghd_data_dir", help="Directory of dghd data", type=str, required=True)
    parser.add_argument("--ethos_data_dir", help="Directory of ethos data", type=str, required=True)
    parser.add_argument("--toxigen_data_dir", help="Directory of toxic gen data", type=str, required=True)


    arguments = parser.parse_args()
    targs = TrainArgs(arguments)
    run_train_dynamics(targs)