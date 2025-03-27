#!/usr/bin/env python3
"""
    load and preprocess the dataset
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
        loding class for the dataset
    """
    def __init__(self):
        """
            class init
        """
        self.data_train, self.data_valid =\
            tfds.load('ted_hrlr_translate/pt_to_en',
                      split=['train', 'validation'],
                      as_supervised=True)
        self.tokenizer_en, self.tokenizer_pt = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
            subword tokenizer 

        :param data: tf.data.Dataset, tuple (pt,en)
            pt: tf.Tensor portuguese sentence
            en: tf.Tensor english sentence
            max vocab size : 2**15

        :return: tokenizer_pt, tokenizer_en
            portuguese and english tokenizer
        """
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2 ** 15
        )
        return self.tokenizer_pt, self.tokenizer_en
