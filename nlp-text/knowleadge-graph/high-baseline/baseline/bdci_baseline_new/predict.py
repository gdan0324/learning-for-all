from model import GRTE
from transformers import WEIGHTS_NAME
from bert4keras.tokenizers import Tokenizer
from util import *
import os
import torch
from transformers import BertConfig
import json
import argparse

def predict_data(batch_ex, batch_spo, f):
    assert len(batch_ex) == len(batch_spo)

    for i in range(len(batch_ex)):
        sample = {}
        spo_list = []
        ex = batch_ex[i]
        spo = batch_spo[i]
        sample['ID'] = ex['id']
        sample['text'] = ex['text']

        for s in spo:
            h = {}
            t = {}

            h['name'] = s[0][-1]
            h['pos'] = [s[0][0], s[0][1]]
            t['name'] = s[2][-1]
            t['pos'] = [s[2][0], s[2][1]]

            spo_list.append({'h':h, 't':t, 'relation':s[1]})

        sample['spo_list'] = spo_list
        s = json.dumps(sample, ensure_ascii=False)

        f.write(s + '\n')

def correct_id(args, test_pred_path):
    output_path = os.path.join(args.output_path)
    test_result_path = os.path.join(output_path, "evalResult.json")

    lines_new = {}
    with open(test_pred_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open(test_result_path, 'w', encoding='utf-8') as fw:
            for line in lines:
                line = json.loads(line.strip('\n'))
                # print(line)
                sample_id = line['ID'].split('_')[0]
                if sample_id not in lines_new:
                    lines_new[sample_id] = []
                    lines_new[sample_id].append(line)
                else:
                    lines_new[sample_id].append(line)

            for key, value in lines_new.items():
                if len(value) > 1:
                    text = ''
                    sample = {}
                    spo_lists = []
                    sample['ID'] = key
                    for i in range(len(value)):
                        text += value[i]['text']

                        total_len = len(text)
                        current_len = len(value[i]['text'])
                        for s in value[i]['spo_list']:
                            h = {}
                            t = {}

                            h['name'] = s['h']['name']
                            h['pos'] = [s['h']['pos'][0] + (total_len - current_len), s['h']['pos'][1] + (total_len - current_len)]
                            t['name'] = s['t']['name']
                            t['pos'] = [s['t']['pos'][0] + (total_len - current_len), s['t']['pos'][1] + (total_len - current_len)]

                            spo_lists.append({'h': h, 't': t, 'relation': s['relation']})

                    print(spo_lists)

                    sample['text'] = text
                    sample['spo_list'] = spo_lists
                    result = json.dumps(sample, ensure_ascii=False)
                    fw.write(result + '\n')
                else:
                    result = json.dumps(value[0], ensure_ascii=False)
                    fw.write(result + '\n')


def evaluate_test(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path):
    f = open(test_pred_path, 'w', encoding='utf-8')

    for batch in test_dataloader:

        batch_ex=batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch

        batch_spo=extract_spo_list(args, tokenizer, id2predicate,id2label,label2id, train_model, batch_ex,batch_token_ids, batch_mask)
        predict_data(batch_ex, batch_spo, f)

    f.close()

def predict():
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] =args.cuda_id

    output_path=os.path.join(args.output_path)
    test_path=os.path.join(args.base_path,args.dataset,"test.json")
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    test_pred_path = os.path.join(output_path, "test_pred.json")

    #label
    label_list=["N/A","SMH","SMT","SS","MMH","MMT","MSH","MST"]

    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i

    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))

    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p=len(id2predicate)
    config.num_label=len(label_list)
    config.rounds=args.rounds
    config.fix_bert_embeddings=args.fix_bert_embeddings

    train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print_config(args)

    test_dataloader=data_generator(args,test_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)

    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    evaluate_test(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path)
    correct_id(args, test_pred_path)
    # print('result')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--cuda_id', default="1", type=str)
    parser.add_argument('--dataset', default='bdci', type=str)
    parser.add_argument('--rounds', default=4, type=int)

    parser.add_argument('--test_batch_size', default=3, type=int)
    parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
    parser.add_argument('--bert_vocab_path', default="./pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/vocab.txt", type=str)
    parser.add_argument('--bert_config_path', default="./pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/config.json", type=str)
    parser.add_argument('--bert_model_path', default="./pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/pytorch_model.bin", type=str)
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--base_path', default="./dataset", type=str)
    parser.add_argument('--output_path', default="output", type=str)

    args = parser.parse_args()

    predict()