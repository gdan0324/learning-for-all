import os

if __name__ == '__main__':
    os.makedirs(os.path.join("train", "ImageSets", "Main"), exist_ok=True)
    files = os.listdir(os.path.join("train", "Annotations"))
    with open(os.path.join("train", "ImageSets", "Main", "train.txt"), 'w') as f:
        for file in files:
            f.write(file.split('.')[0])
            f.write('\n')
