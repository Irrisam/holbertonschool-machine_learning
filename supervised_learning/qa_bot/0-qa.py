#!/usr/bin/env python3
"""
    module query response
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def retrieve_answer(query, context):
    """
        finds a text segment in a given reference
        to answer a question

    :param query: string, the question you're asking
    :param context: string, the doc where we search for the answer

    :return: string, extracted answer
        if no answer found: None
    """

    # load tokenizer & model
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word'
                                              '-masking-finetuned-squad')
    model = (
        hub.load("https://www.kaggle.com/models/seesee/bert/"
                 "TensorFlow2/uncased-tf2-qa/1"))

    # tokenize query & context
    q_tokens = tokenizer.tokenize(query)
    ctx_tokens = tokenizer.tokenize(context)

    ###########################
    #   token preprocessing   #
    ###########################

    # combine tokens with special markers
    token_sequence = ['[CLS]'] + q_tokens + ['[SEP]'] + ctx_tokens + ['[SEP]']

    # convert tokens to their numerical IDs
    input_word_ids = tokenizer.convert_tokens_to_ids(token_sequence)

    # create tensor mask: 1 means relevant token
    input_mask = [1] * len(input_word_ids)
    # distinguish between query and context
    input_type_ids = ([0] * (1 + len(q_tokens) + 1) +
                      [1] * (len(ctx_tokens) + 1))

    # convert to TensorFlow tensors
    input_word_ids, input_mask, input_type_ids = (
        map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t,
                                 dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_type_ids)))

    ###########################
    #  bert model inference   #
    ###########################

    # run model inference
    predictions = model([input_word_ids, input_mask, input_type_ids])
    # predictions show probabilities for start & end of the answer

    ###########################
    #    extract response     #
    ###########################

    # find the most likely starting token (+1 to skip CLS token)
    start_idx = tf.argmax(predictions[0][0][1:]) + 1
    # find the most likely ending token (+1 to skip CLS token)
    end_idx = tf.argmax(predictions[1][0][1:]) + 1
    # grab tokens from start to end for the answer
    extracted_tokens = token_sequence[start_idx: end_idx + 1]
    if not extracted_tokens:
        return None

    # convert the extracted tokens to text
    response_text = tokenizer.convert_tokens_to_string(extracted_tokens)

    return response_text
