import os

if __name__ == '__main__':
    files = os.listdir(os.path.join("train", "JPEGImages"))
    for file in files:
        os.rename("train/JPEGImages/%s" % file, "train/JPEGImages/%s.jpg" % file.split('.')[0])
