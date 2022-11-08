import os

# 修改路径
OUT_PATH = 'C:\\Users\\96212\\Desktop\\labels'
alltest_path = 'C:\\Users\\96212\\Desktop\\汇报\\科大讯飞赛题\\小样本\\data\\testjpg'
##########


file_test_lists = [os.path.splitext(img)[0] for img in os.listdir(alltest_path)]
out_lists = [os.path.splitext(txt)[0] for txt in os.listdir(OUT_PATH)]

for fileitem in file_test_lists:
    if fileitem not in out_lists:
        # 正常图片直接给空
        emptyfile = fileitem + '.txt'
        with open(os.path.join(OUT_PATH, emptyfile), 'w') as file:
            pass
print('finish')
