#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf
"""
Is that any better
"""


class Dataset:
    """
     machine translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and validation
        datasets.
        """
        # Load the Portuguese to English translation dataset
        data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='train', as_supervised=True)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation', as_supervised=True)

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)

        # Tokenize the datasets
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers and adapts them to
        the dataset.

        :param data: tf.data.Dataset containing tuples of (pt, en) sentences.

        Returns:
        - :tokenizer_pt: Portuguese.
        - :tokenizer_en: English.
        """
        # Get and decode sentences from the dataset (build iterator)
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Load the pre-trained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train both tokenizers on the dataset sentence iterators
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        # Update the Dataset tokenizers with the newly trained ones
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        :param pt: tf.Tensor containing the Portuguese sentence
        :param en: tf.Tensor containing the corresponding English sentence

        Returns:
        - pt_tokens: np.ndarray containing the Portuguese tokens
        - en_tokens: np.ndarray containing the English tokens
        """
        # Decode the tensors to strings
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Get vocab size for start/end tokens
        vocab_size_pt = len(self.tokenizer_pt.vocab)
        vocab_size_en = len(self.tokenizer_en.vocab)

        # Define start and end tokens
        start_token_pt = vocab_size_pt
        end_token_pt = vocab_size_pt + 1
        start_token_en = vocab_size_en
        end_token_en = vocab_size_en + 1

        # Tokenize the sentences (without special tokens)
        pt_tokens = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        # Add start and end tokens
        pt_tokens = [start_token_pt] + pt_tokens + [end_token_pt]
        en_tokens = [start_token_en] + en_tokens + [end_token_en]

        # Convert to numpy arrays using tf.constant().numpy()
        pt_tokens = tf.constant(pt_tokens).numpy()
        en_tokens = tf.constant(en_tokens).numpy()

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method.

        :param pt: tf.Tensor containing the Portuguese sentence
        :param en: tf.Tensor containing the corresponding English sentence

        Returns:
        - pt_tokens: tf.Tensor containing the Portuguese tokens
        - en_tokens: tf.Tensor containing the English tokens
        """
        # Use tf.py_function to wrap the encode method
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set the shape of the tensors (unknown sequence length)
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
