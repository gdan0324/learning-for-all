"""
Time:2022/10/5 10:25
Author:ECCUSYB
"""
import os
import clip
import torch
from PIL import Image

img_pah = '2TWS89CXZV.jpg'
classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# 准备输入集
image = Image.open(img_pah)
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)  # 生成文字描述

# 特征编码
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# 选取参数最高的标签
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # 对图像描述和图像特征
values, indices = similarity[0].topk(1)

# 输出结果
print("\nTop predictions:\n")
print('classes:{} score:{:.2f}'.format(classes[indices.item()], values.item()))
