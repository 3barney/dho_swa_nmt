import torch

from typing import Dict, List, Any
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from character_vocabulary import CharacterVocabulary
from transformers import (AutoTokenizer)

SRC_LANG_SHORT = "luo"
TGT_LANG_SHORT = "swa"

class NMTCharSubwordDataset(TorchDataset):
    def __init__(self,
                 hf_dataset: HFDataset,
                 subword_tokenizer: AutoTokenizer,
                 char_vocab: CharacterVocabulary,
                 max_subword_seq_len: int,
                 max_char_len_per_subword: int,
                 src_lang_key: str = SRC_LANG_SHORT,
                 tgt_lang_key: str = TGT_LANG_SHORT):
        self.hf_dataset = hf_dataset
        self.subword_tokenizer = subword_tokenizer
        self.char_vocab = char_vocab
        self.max_subword_seq_len = max_subword_seq_len
        self.max_char_len_per_subword = max_char_len_per_subword
        self.src_lang_key = src_lang_key
        self.tgt_lang_key = tgt_lang_key

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.hf_dataset[idx]
        src_text = example[self.src_lang_key]
        tgt_text = example[self.tgt_lang_key]

        # Subword tokenization for source
        # Important: NLLB tokenizer might add lang codes automatically.
        # For CharCNN input, we want the characters of the *actual text subwords*.
        # We'll rely on the main model to handle lang codes if it's NLLB.
        # If student is mT5, it also has its own way of handling languages or multitask.
        # The subword_tokenizer for CharCNN should ideally NOT add language codes,
        # or we need to strip them if convert_ids_to_tokens includes them.
        # For simplicity here, we assume the tokenizer used for char processing
        # gives clean subwords.
        src_subword_tokenized = self.subword_tokenizer(
            src_text,
            max_length=self.max_subword_seq_len,
            truncation=True,
            padding=False  # Pad in collate_fn
        )
        src_subword_ids = src_subword_tokenized['input_ids']
        src_attention_mask = src_subword_tokenized['attention_mask']

        # Character representation for source subwords
        # `convert_ids_to_tokens` might add special prefixes (like  for SentencePiece)
        # or decode special tokens like </s>. These should be part of char_vocab.
        src_subword_strings = self.subword_tokenizer.convert_ids_to_tokens(src_subword_ids, skip_special_tokens=False)

        src_char_ids_for_subwords = []
        for subword_str in src_subword_strings:
            # Handle tokenizer-specific decoding artifacts if necessary.
            # For NLLB, convert_ids_to_tokens might return things like '<s>', '</s>', 'luo_Latn'
            # These need to be correctly represented by characters or handled.
            # The char_vocab.encode_subword is designed to take these strings.
            char_ids = self.char_vocab.encode_subword(subword_str, self.max_char_len_per_subword)
            src_char_ids_for_subwords.append(torch.tensor(char_ids, dtype=torch.long))

        # Subword tokenization for target (labels)
        # For NLLB, target text needs the target language code as prefix during training by the model itself.
        # The tokenizer for `labels` should prepare it as NLLB expects it.
        # The `DataCollatorForSeq2Seq` usually handles shifting labels for decoder input ids.
        # If not using it, we need to be careful. Here, we just tokenize.
        tgt_subword_tokenized = self.subword_tokenizer(
            text_target=tgt_text,  # Use text_target for mT5/NLLB style models
            max_length=self.max_subword_seq_len,
            truncation=True,
            padding=False  # Pad in collate_fn
        )
        labels = tgt_subword_tokenized['input_ids']

        return {
            "input_ids": torch.tensor(src_subword_ids, dtype=torch.long),
            "attention_mask": torch.tensor(src_attention_mask, dtype=torch.long),
            "source_char_ids": src_char_ids_for_subwords,  # List of tensors, will be padded by collate_fn
            "labels": torch.tensor(labels, dtype=torch.long),
        }