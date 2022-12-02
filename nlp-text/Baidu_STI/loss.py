"""

"""
#----------------------------- system import ----------------------------------
#----------------------------- paddle import ----------------------------------
import paddle
#----------------------------- customer import ----------------------------------


class CrossEntropyLossForRobust(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForRobust, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(
            input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(
            input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss