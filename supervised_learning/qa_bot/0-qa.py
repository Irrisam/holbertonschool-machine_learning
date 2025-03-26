#!/usr/bin/env python3
"""
    Module Query Response
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def retrieve_answer(query, context):
    """
        Identifies a segment of text within a given reference
        to respond to a posed query

    :param query: string, inquiry needing an answer
    :param context: string, supporting document to locate response

    :return: string, extracted answer
        if no response found: None
    """

    # initialize tokenizer & model
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word'
                                              '-masking-finetuned-squad')
    model = (
        hub.load("https://www.kaggle.com/models/seesee/bert/"
                 "TensorFlow2/uncased-tf2-qa/1"))

    # tokenize question and reference
    q_tokens = tokenizer.tokenize(query)
    ctx_tokens = tokenizer.tokenize(context)

    ###########################
    #   TOKEN PREPROCESSING   #
    ###########################

    # merge tokens with special indicators
    token_sequence = ['[CLS]'] + q_tokens + ['[SEP]'] + ctx_tokens + ['[SEP]']

    # transform tokens into numerical identifiers
    input_word_ids = tokenizer.convert_tokens_to_ids(token_sequence)

    # generate tensor mask: 1 signifies relevant token
    input_mask = [1] * len(input_word_ids)
    # differentiate query from context
    input_type_ids = ([0] * (1 + len(q_tokens) + 1) +
                      [1] * (len(ctx_tokens) + 1))

    # transform into TensorFlow tensors
    input_word_ids, input_mask, input_type_ids = (
        map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t,
                                 dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_type_ids)))

    ###########################
    #  BERT MODEL INFERENCE   #
    ###########################

    # execute model inference
    predictions = model([input_word_ids, input_mask, input_type_ids])
    # predictions indicate probabilities of tokens being start & end
    # of the extracted answer in provided text

    ###########################
    #    RESPONSE EXTRACTION   #
    ###########################

    # determine most probable starting token (+1 to skip CLS token)
    start_idx = tf.argmax(predictions[0][0][1:]) + 1
    # determine most probable ending token (+1 to skip CLS token)
    end_idx = tf.argmax(predictions[1][0][1:]) + 1
    # retrieve token subset corresponding to answer
    extracted_tokens = token_sequence[start_idx: end_idx + 1]
    if not extracted_tokens:
        return None

    # convert extracted tokens into readable text
    response_text = tokenizer.convert_tokens_to_string(extracted_tokens)

    return response_text
