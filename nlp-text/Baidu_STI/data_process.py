""" -------------------system import------------------- """
import os
# import numpy as np
import pandas as pd
import pyarrow as pa
from functools import partial
from datasets import Dataset

""" -------------------paddle import------------------- """
import paddle
# # import paddlenlp
# from paddlenlp.data import DataCollatorWithPadding
# # from paddlenlp.datasets import load_dataset
# from paddlenlp.transformers import AutoTokenizer

""" 
function prepare_train_features and prepare_validation_features are in
PaddleNLP/examples/machine_reading_comprehension/SQuAD/run_squad.py 
examples need to    "load_dataset('dureader_robust')"
than can import these two functions from "utils" directly
"""
# from utils import prepare_train_features, prepare_validation_features

""" ---------------
----customer import------------------- """
from paddlenlp_funcs import prepare_train_features, prepare_validation_features


class DataProcess():
    def __init__(self, data_path, MODEL_NAME):
        """
        params:
        PRE_MODEL_NAME: 'ernie-3.0-medium-zh'
        max_seq_length: 512, Max window length of input setence
        doc_stride: 256, Stride between windows, also means the overlapping
        """
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        max_seq_length = 1024
        doc_stride = 1024
        self.batch_size = 64

        self.train_path = os.path.join(data_path, 'train_data/train.json')
        self.dev_path = os.path.join(data_path, 'dev_data/dev.json')
        self.test_path = os.path.join(data_path, 'test_data/test.json')

        self.train_trans_func = partial(prepare_train_features,
                                        max_seq_length=max_seq_length,
                                        doc_stride=doc_stride,
                                        tokenizer=self.tokenizer)

        self.dev_trans_func = partial(prepare_validation_features,
                                      max_seq_length=max_seq_length,
                                      doc_stride=doc_stride,
                                      tokenizer=self.tokenizer)

        self.test_trans_func = partial(prepare_validation_features,
                                       max_seq_length=max_seq_length,
                                       doc_stride=doc_stride,
                                       tokenizer=self.tokenizer)

        self.__mainprocess__()
        self.__dataloader__()

    def __mainprocess__(self):
        train = pd.read_json(self.train_path, lines=True).reset_index(drop=True)
        dev = pd.read_json(self.dev_path, lines=True).reset_index(drop=True)
        test = pd.read_json(self.test_path, lines=True).reset_index(drop=True)

        train['id'] = [f'train_{id}' for id in range(len(train))]
        dev['id'] = [f'dev_{id}' for id in range(len(dev))]
        test['id'] = [f'test_{id}' for id in range(len(test))]
        test['answer_list'] = [[] for i in range(len(test))]
        test['answer_start_list'] = [[] for i in range(len(test))]

        """
        change train dev test DataFrame into datasets
        """

        train_examples = Dataset(pa.Table.from_pandas(train))
        dev_examples = Dataset(pa.Table.from_pandas(dev))
        test_examples = Dataset(pa.Table.from_pandas(test))

        """
        delete all of the original columns while map the trans function
        """
        column_names = train_examples.column_names
        """
        train_ds keys: 'input_ids', 'token_type_ids', 'start_positions', 'end_positions'
        """
        self.train_ds = train_examples.map(self.train_trans_func,
                                           batched=True,
                                           num_proc=3,
                                           remove_columns=column_names)
        self.train_ds = self.train_ds.remove_columns(['id'])

        """
        dev_ds keys: 'offset_mapping', 'input_ids', 'token_type_ids', 'example_id'
        """
        self.dev_ds = dev_examples.map(self.dev_trans_func,
                                       batched=True,
                                       num_proc=3,
                                       remove_columns=column_names)

        self.dev_ds_for_model = self.dev_ds.remove_columns(['example_id', 'offset_mapping'])

        """
        test_ds keys: 'offset_mapping', 'input_ids', 'token_type_ids', 'example_id'
        """
        column_names.remove('org_answer')
        self.test_ds = test_examples.map(self.dev_trans_func,
                                         batched=True,
                                         num_proc=3,
                                         remove_columns=column_names)
        self.test_ds_for_model = self.test_ds.remove_columns(["example_id", "offset_mapping"])

        self.train_examples = train_examples
        self.dev_examples = dev_examples
        self.test_examples = test_examples

    def __dataloader__(self):
        train_batch_sampler = paddle.io.DistributedBatchSampler(self.train_ds, batch_size=self.batch_size, shuffle=True)
        dev_batch_sampler = paddle.io.BatchSampler(self.dev_ds, batch_size=self.batch_size, shuffle=True)
        """ test batch size is 16 """
        test_batch_sampler = paddle.io.BatchSampler(self.test_ds, batch_size=int(self.batch_size / 4), shuffle=False)

        train_batchify_fn = DataCollatorWithPadding(self.tokenizer)
        dev_batchify_fn = DataCollatorWithPadding(self.tokenizer)
        test_batchify_fn = DataCollatorWithPadding(self.tokenizer)

        self.train_data_loader = paddle.io.DataLoader(dataset=self.train_ds,
                                                      batch_sampler=train_batch_sampler,
                                                      collate_fn=train_batchify_fn,
                                                      return_list=True)

        self.dev_data_loader = paddle.io.DataLoader(dataset=self.dev_ds,
                                                    batch_sampler=dev_batch_sampler,
                                                    collate_fn=dev_batchify_fn,
                                                    return_list=True)

        self.test_data_loader = paddle.io.DataLoader(dataset=self.test_ds_for_model,
                                                     batch_sampler=test_batch_sampler,
                                                     collate_fn=test_batchify_fn,
                                                     return_list=True)


if __name__ == '__main__':
    data_path = r'./data'
    MODEL_NAME = 'ernie-3.0-medium-zh'
    data_process = DataProcess(data_path, MODEL_NAME)
    train_data_loader = data_process.train_data_loader
    dev_data_loader = data_process.dev_data_loader
    print('check data loader type&num', type(train_data_loader), type(dev_data_loader))
