import copy
import json


def get_spos(dic_single, spo_list, text_count):
    dic_single_copy = copy.deepcopy(dic_single)
    for spo in spo_list:
        h = spo['h']
        t = spo['t']
        relation = spo['relation']

        arr_h = []
        arr_h.append(h['pos'][0])
        arr_h.append(h['pos'][1])
        arr_h.append(h['name'])

        arr_t = []
        arr_t.append(t['pos'][0])
        arr_t.append(t['pos'][1])
        arr_t.append(t['name'])

        arr_spo = []
        arr_spo.append(arr_h)
        arr_spo.append(relation)
        arr_spo.append(arr_t)
        dic_single_copy['spos'].append((arr_spo))

    spos_new = sorted(dic_single_copy['spos'], key=lambda x: x[0])
    spos_new = sorted(spos_new, key=lambda x: x[2])

    spos_new1 = []
    s_idx = 0
    e_idx = 0

    for s in text_count[:-1]:
        s_idx += len(s)
    for e in text_count:
        e_idx += len(e)

    # print(spos_new)
    for spo in spos_new:
        # print(spo)
        if spo[0][0] >= s_idx and spo[-1][1] <= e_idx:
            spo[0][0] = spo[0][0] - s_idx
            spo[0][1] = spo[0][1] - s_idx
            spo[2][0] = spo[2][0] - s_idx
            spo[2][1] = spo[2][1] - s_idx
            spos_new1.append(spo)
        else:
            continue

    # print(spos_new1)
    # exit()
    return spos_new1


def train_generator(file_train_bdci, file_train):
    with open(file_train_bdci, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        result_arr = []
        with open(file_train, 'w', encoding='utf-8') as fw:
            for line in lines:
                line = line.strip()
                if line == "":
                    continue

                dic_single = {}
                text_count = []

                line = json.loads(line)
                line_id = line['ID']
                line_text = line['text']
                spo_list = line['spo_list']

                dic_single['id'] = line_id
                dic_single['text'] = line_text
                dic_single['spos'] = []

                if line_text in result_arr:
                    continue

                if len(line_text) > 200:
                    line_arr = []
                    text = ''

                    for l in range(len(line_text)):
                        text += line_text[l]
                        if line_text[l] in ['，', '。', '！', '？', '、']:
                            line_arr.append(text)
                            text = ''
                        if l == len(line_text) - 1 and text != line_arr[-1]:
                            line_arr.append(text)

                    text_new = ''
                    n = 0
                    for i in range(len(line_arr)):
                        dic_single_new = {}
                        text_original = text_new
                        text_new += line_arr[i]
                        if len(text_new) <= 200:
                            if i == len(line_arr) - 1:
                                # id_new = line_id + '_' + str(n)
                                # out = {'id': id_new, 'text': text_new}
                                text_count.append(text_new)
                                spos_new1 = get_spos(dic_single, spo_list, text_count)
                                dic_single_new['id'] = line_id
                                dic_single_new['text'] = text_new
                                dic_single_new['spos'] = spos_new1
                                result_arr.append(dic_single_new)
                            else:
                                continue
                        else:
                            # id_new = line_id + '_' + str(n)
                            # out = {'id': id_new, 'text': text_original}
                            text_count.append(text_original)
                            spos_new1 = get_spos(dic_single, spo_list, text_count)
                            dic_single_new['id'] = line_id
                            dic_single_new['text'] = text_original
                            dic_single_new['spos'] = spos_new1
                            result_arr.append(dic_single_new)
                            text_new = line_arr[i]
                            n += 1
                            if i == len(line_arr) - 1:
                                # id_new = line_id + '_' + str(n)
                                # out = {'id': id_new, 'text': text_new}
                                text_count.append(text_new)
                                spos_new1 = get_spos(dic_single, spo_list, text_count)
                                dic_single_new['id'] = line_id
                                dic_single_new['text'] = text_new
                                dic_single_new['spos'] = spos_new1
                                result_arr.append(dic_single_new)
                else:
                    for spo in spo_list:
                        h = spo['h']
                        t = spo['t']
                        relation = spo['relation']

                        arr_h = []
                        arr_h.append(h['pos'][0])
                        arr_h.append(h['pos'][1])
                        arr_h.append(h['name'])

                        arr_t = []
                        arr_t.append(t['pos'][0])
                        arr_t.append(t['pos'][1])
                        arr_t.append(t['name'])

                        arr_spo = []
                        arr_spo.append(arr_h)
                        arr_spo.append(relation)
                        arr_spo.append(arr_t)
                        dic_single['spos'].append(arr_spo)

                    result_arr.append(dic_single)

                # print(result_arr[0])
                # print('============')
                # exit()

            print('train:', len(result_arr))
            result_json = json.dumps(result_arr, ensure_ascii=False, indent=2)
            fw.write(result_json)


def test_generator(file_evalA, file_test):
    with open(file_evalA, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        result_arr = []

        with open(file_test, 'w', encoding='utf-8') as fw:
            for line in lines:
                line = json.loads(line)
                line_id = line['ID']
                line_text = line['text']

                if len(line_text) > 200:
                    line_arr = []
                    text = ''
                    for l in range(len(line_text)):
                        text += line_text[l]
                        if line_text[l] in ['，', '。', '！', '？', '、']:
                            line_arr.append(text)
                            text = ''
                        if l == len(line_text) - 1 and text != line_arr[-1]:
                            line_arr.append(text)

                    text_new = ''
                    n = 0
                    for i in range(len(line_arr)):
                        text_original = text_new
                        text_new += line_arr[i]
                        if len(text_new) <= 200:
                            if i == len(line_arr) - 1:
                                id_new = line_id + '_' + str(n)
                                out = {'id': id_new, 'text': text_new}
                                result_arr.append(out)
                            else:
                                continue
                        else:
                            id_new = line_id + '_' + str(n)
                            out = {'id': id_new, 'text': text_original}
                            result_arr.append(out)
                            text_new = line_arr[i]
                            n += 1
                            if i == len(line_arr) - 1:
                                id_new = line_id + '_' + str(n)
                                out = {'id': id_new, 'text': text_new}
                                result_arr.append(out)
                else:
                    out = {'id': line_id, 'text': line_text}
                    result_arr.append(out)

            print('test:', len(result_arr))
            result_json = json.dumps(result_arr, ensure_ascii=False, indent=2)
            fw.write(result_json)


if __name__ == '__main__':
    file_train_bdci = 'dataset/bdci/train_bdci.json'
    file_train = 'dataset/bdci/train.json'
    file_evalA = 'dataset/bdci/evalA.json'
    file_test = 'dataset/bdci/test.json'

    train_generator(file_train_bdci, file_train)
    test_generator(file_evalA, file_test)
