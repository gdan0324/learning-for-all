进入ernie_dqa_task1/applications/tasks/sequence_labeling目录
train: sh begin_train.sh
inference: python run_infer.py --param_path=./examples/seqlab_ernie_fc_ch_infer.json
配置文件位于examples目录下
训练好的模型参数保存在output目录下
evaluation: sh evaluate.sh
