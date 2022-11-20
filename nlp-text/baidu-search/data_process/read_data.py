"""
Time:2022/11/10 22:44
Author:ECCUSYB
"""
import json


def main():
    filename = "../data/data_task1/dev_data/dev.json"
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()