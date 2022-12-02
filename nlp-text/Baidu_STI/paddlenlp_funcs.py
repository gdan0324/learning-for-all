"""
This file save all of the functions collected from paddlenlp

"""


def prepare_train_features(examples,tokenizer,doc_stride,max_seq_length):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # contexts = [examples[i]['context'] for i in range(len(examples))]
    # questions = [examples[i]['question'] for i in range(len(examples))]
    contexts = examples["doc_text"]
    questions = examples["query"]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["id"] = []
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    # for i in range(len(tokenized_examples["input_ids"])):
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offsets = tokenized_examples['offset_mapping'][i]

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples['token_type_ids'][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        # sample_index = tokenized_examples['overflow_to_sample'][i]
        sample_index = sample_mapping[i]

        # answers = examples['answers'][sample_index]
        # # answer_starts = examples[sample_index]['answer_starts']
        # answer_starts = answers["answer_start"]
        # # Start/end character index of the answer in the text.
        # start_char = answer_starts[0]
        # # end_char = start_char + len(answers[0])
        # end_char = start_char + len(answers["text"][0])

        # answers = examples['answers'][sample_index]
        answer_starts = examples['answer_start_list'][sample_index]

        if len(answer_starts)<1:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["id"].append(examples['id'][sample_index])
        else:
            # answer_starts = answers["answer_start"]
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            # end_char = start_char + len(answers[0])
            end_char = start_char + len( examples["answer_list"][sample_index][0])



            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Minus one more to reach actual text
            token_end_index -= 1
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append( token_end_index + 1)
            tokenized_examples["id"].append(examples['id'][sample_index])
    return tokenized_examples


def prepare_validation_features(examples, tokenizer, doc_stride, max_seq_length):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = examples['doc_text']
    questions = examples["query"]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length)
    tokenized_examples["example_id"] = []
    sample_mapping = tokenized_examples.pop("overflow_to_sample")
    # For validation, there is no need to compute start and end positions
    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples['token_type_ids'][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        # sample_index = tokenized_examples['overflow_to_sample'][i]
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples['id'][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples