""" -------------------system import------------------- """
# import sys
# sys.path.append('../Baidu_STI')
# import pandas as pd
# from datasets import Dataset

""" -------------------paddle import------------------- """
# import paddle
# import paddlenlp
# from paddlenlp.transformers import AutoModelForQuestionAnswering


""" -------------------customer import------------------- """
# from data_process import DataProcess
from loss import CrossEntropyLossForRobust
from metrix_compute import evaluate


class Train():
    def __init__(self, train_data_loader, MODEL_NAME):
        self.train_data_loader = train_data_loader
        self.model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        learning_rate = 3e-5
        self.epochs = 1
        warmup_proportion = 0.1
        weight_decay = 0.01
        num_training_steps = len(train_data_loader) * self.epochs
        self.lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]


        self.optimizer = paddle.optimizer.AdamW(
            learning_rate = self.lr_scheduler,
            parameters = self.model.parameters(),
            weight_decay = weight_decay,
            apply_decay_param_fun = lambda x: x in decay_params
        )

        self.criterion = CrossEntropyLossForRobust()



    def train(self):
        global_step = 0
        for epoch in range(1, self.epochs+1):
            for step, batch in enumerate(self.train_data_loader, start=1):
                global_step += 1
                input_ids, segment_ids, start_positions, end_positions = batch
                logits = self.model(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"])
                loss = self.criterion(logits, (batch["start_positions"], batch["end_positions"]))

                if global_step % 400 == 0 :
                    print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_grad()
        return


if __name__ == '__main__':
    data_path = r'./data'
    data_path_test_result = './subtask1_test_pred.txt'
    MODEL_NAME = 'ernie-3.0-medium-zh'
    """ dataloader """
    data_process = DataProcess(data_path, MODEL_NAME)

    """ train """
    train_process = Train(data_process.train_data_loader, MODEL_NAME)
    train_process.train()

    """ evaluation """
    dev_all_predictions, dev_all_nbest_json, dev_scores_diff_json = evaluate(model=train_process.model,
                                                                    raw_dataset=data_process.dev_examples,
                                                                    dataset=data_process.dev_ds,
                                                                    data_loader=data_process.dev_data_loader,
                                                                    is_test=False)

    """ test and write into txt files"""
    test_all_predictions, test_all_nbest_json, test_scores_diff_json=evaluate(model=train_process.model,
                                                                            raw_dataset=data_process.test_examples,
                                                                            dataset=data_process.test_ds,
                                                                            data_loader=data_process.test_data_loader,
                                                                            is_test=True)

    sub_data=[]

    for i in range(len(data_process.test_examples["query"])):
        prob=test_all_nbest_json[data_process.test_examples['id'][i]][0]['probability']
        text=test_all_nbest_json[data_process.test_examples['id'][i]][0]['text']
        sub_data.append([prob,text])
    sub=pd.DataFrame(sub_data,columns=['prob','text'])
    sub.to_csv(data_path_test_result ,sep='\t',header=None,index=None)
    sub['text_len']=sub['text'].map(len)
    sub['text']=sub['text'].apply(lambda  x:"NoAnswer" if len(x.strip())<2 else x)
    sub[['prob','text']].to_csv('subtask1_test_pred.txt',sep='\t',header=None,index=None)





