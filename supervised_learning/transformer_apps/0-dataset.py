#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers
'''exercice about SIMPLY loading a dataset.... '''


class Dataset:
    """Dataset class for Portuguese to English machine translation."""

    def __init__(self):
        """Initialize the dataset with train/validation splits and tokenizers."""
        # Load train and validation splits separately
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # Create tokenizers from the training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset with examples as tuples (pt, en)
                  - pt: tf.Tensor containing Portuguese sentence
                  - en: tf.Tensor containing English sentence

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Load pre-trained tokenizers using transformers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        # Extract text data for training tokenizers
        pt_texts = []
        en_texts = []

        for pt, en in data:
            # Decode tf.Tensor to string
            pt_texts.append(pt.numpy().decode('utf-8'))
            en_texts.append(en.numpy().decode('utf-8'))

        # Set max_length to handle sequences properly
        tokenizer_pt.model_max_length = 512
        tokenizer_en.model_max_length = 512

        return tokenizer_pt, tokenizer_en
