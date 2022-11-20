import argparse
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer
from model import GRTE
from util import *
from tqdm import tqdm
import os
import torch.nn as nn
import torch
from transformers import BertConfig
import json
import random


def train():
    set_seed()
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    output_path = os.path.join(args.output_path)
    train_path = os.path.join(args.base_path, args.dataset, "train.json")
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    dev_pred_path = os.path.join(output_path, "dev_pred.json")
    log_path = os.path.join(output_path, "log.txt")

    # label
    label_list = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH", "MST"]

    id2label, label2id = {}, {}
    for i, l in enumerate(label_list):
        id2label[str(i)] = l
        label2id[l] = i

    train_data = json.load(open(train_path, encoding='utf-8'))
    all_data = np.array(train_data)
    random.shuffle(all_data)
    num = len(all_data)
    split_num = int(num * 0.8)
    train_data = all_data[: split_num]
    dev_data = all_data[split_num:]

    id2predicate, predicate2id = json.load(open(rel2id_path, encoding='utf-8'))

    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    config.num_label = len(label_list)
    config.rounds = args.rounds
    config.fix_bert_embeddings = args.fix_bert_embeddings

    train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to("cuda")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print_config(args)

    dataloader = data_generator(args, train_data, tokenizer, [predicate2id, id2predicate], [label2id, id2label],
                                args.batch_size, random=True)
    dev_dataloader = data_generator(args, dev_data, tokenizer, [predicate2id, id2predicate], [label2id, id2label],
                                    args.val_batch_size, random=False, is_train=False)

    t_total = len(dataloader) * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )

    best_f1 = -1.0
    step = 0
    crossentropy = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(args.num_train_epochs):
        train_model.train()
        epoch_loss = 0
        print('current epoch: %d\n' % (epoch))
        with tqdm(total=dataloader.__len__(), desc="train") as t:
            for i, batch in enumerate(dataloader):
                batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
                batch_token_ids, batch_mask, batch_label, batch_mask_label = batch
                table = train_model(batch_token_ids, batch_mask)
                table = table.reshape([-1, len(label_list)])
                batch_label = batch_label.reshape([-1])

                loss = crossentropy(table, batch_label.long())
                loss = (loss * batch_mask_label.reshape([-1])).sum()

                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                train_model.zero_grad()

                t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                t.update(1)

        f1, precision, recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model, dev_dataloader,
                                         dev_pred_path)

        if f1 > best_f1:
            # Save model checkpoint
            best_f1 = f1
            torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))

        epoch_loss = epoch_loss / dataloader.__len__()
        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f\t" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)

    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model, dev_dataloader,
                                     dev_pred_path)
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)


def evaluate(args, tokenizer, id2predicate, id2label, label2id, model, dataloader, evl_path):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm()
    for batch in dataloader:

        batch_ex = batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch

        batch_spo = extract_spo_list(args, tokenizer, id2predicate, id2label, label2id, model, batch_ex,
                                     batch_token_ids, batch_mask)
        for i, ex in enumerate(batch_ex):
            one = batch_spo[i]
            R = set([(tuple(item[0]), item[1], tuple(item[2])) for item in one])
            T = set([(tuple(item[0]), item[1], tuple(item[2])) for item in ex['spos']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': ex['text'],
                'spos_list': list(T),
                'spos_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(s + '\n')
    pbar.close()
    f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--cuda_id', default="2", type=str)
    parser.add_argument('--dataset', default='bdci', type=str)
    parser.add_argument('--rounds', default=4, type=int)
    parser.add_argument('--train', default="train", type=str)

    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--val_batch_size', default=4, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
    parser.add_argument('--bert_vocab_path',
                        default="./pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/vocab.txt", type=str)
    parser.add_argument('--bert_config_path',
                        default="./pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/config.json", type=str)
    parser.add_argument('--bert_model_path',
                        default="./pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/pytorch_model.bin",
                        type=str)
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--min_num', default=1e-7, type=float)
    parser.add_argument('--base_path', default="./dataset", type=str)
    parser.add_argument('--output_path', default="output", type=str)

    args = parser.parse_args()
    train()
