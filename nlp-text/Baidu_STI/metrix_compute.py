# ----------------------------- system import ----------------------------------
import time

# ----------------------------- paddle import ----------------------------------
import paddle
from paddlenlp.metrics.squad import compute_prediction


# from paddlenlp.metrics.squad import squad_evaluate

@paddle.no_grad()
def evaluate(model, raw_dataset, dataset, data_loader, is_test=False):
    model.eval()
    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()
    for batch in data_loader:
        start_logits_tensor, end_logits_tensor = model(batch["input_ids"],
                                                       batch["token_type_ids"])
        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])
    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        raw_dataset, dataset,
        (all_start_logits, all_end_logits), False, 20, 256)

    # if is_test:
    #     # Can also write all_nbest_json and scores_diff_json files if needed
    #     with open('prediction.json', "w", encoding='utf-8') as writer:
    #         writer.write(
    #             json.dumps(
    #                 all_predictions, ensure_ascii=False, indent=4) + "\n")
    # else:
    #     squad_evaluate(
    #         examples=[raw_data for raw_data in raw_dataset],
    #         preds=all_predictions,
    #         is_whitespace_splited=False)

    for i in range(len(raw_dataset["query"])):
        print()
        print('问题：', raw_dataset['query'][i])
        print('原文：', ''.join(raw_dataset['doc_text'][i]))
        print('答案：', all_predictions[raw_dataset['id'][i]])
        if i >= 5:
            break
    return all_predictions, all_nbest_json, scores_diff_json
    # model.train()
