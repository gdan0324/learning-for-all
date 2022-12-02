

# 1. 本地能Debug的环境

​	conda create -n paddle python=3.7
​	pip install paddlenlp==2.2.5
​	pip install paddlepaddle==2.3.2

# 2. 增加 seqlab_ernie_fc_ch_cpu.json 参数文件

​	放在 applications/tasks/sequence_labeling/examples/seqlab_ernie_fc_ch_cpu.json 主要是因为本地调试可能只能用cpu。

	与 seqlab_ernie_fc_ch.json 的区别是：
	train_reader 里面的 config的 batch_size由32改成了2
	trainer里面的 PADDLE_PLACE_TYPE 改成 cpu、train_log_step 由100改成2、eval_step由100改成4、

​	 batch设置成2，step设置成2，就是前进 2 * 2 个打印一次

