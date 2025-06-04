import torch

from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

MAX_CHAR_LEN_PER_SUBWORD = 16

class CustomDataCollator:
    def __init__(self, student_tokenizer: AutoTokenizer, char_pad_token_id: int, max_char_len_per_subword: int):
        self.student_tokenizer = student_tokenizer  # Needed for pad_token_id
        self.char_pad_token_id = char_pad_token_id
        self.max_char_len_per_subword = max_char_len_per_subword

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:  # Return Any for non-tensor fields
        # Student inputs
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        source_char_ids_list_of_lists = [item["source_char_ids"] for item in batch]  # list of list of tensors

        padded_input_ids = pad_sequence(input_ids_list, batch_first=True,
                                        padding_value=self.student_tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # Labels for student
        padded_labels = None
        if "labels" in batch[0] and batch[0]["labels"] is not None:
            labels_list = [item["labels"] for item in batch]
            padded_labels = pad_sequence(labels_list, batch_first=True,
                                         padding_value=-100)  # Use -100 for ignored labels

        # Character IDs padding
        batch_size = len(batch)
        max_subwords_in_batch = padded_input_ids.size(1)
        padded_source_char_ids = torch.full(
            (batch_size, max_subwords_in_batch, self.max_char_len_per_subword),
            fill_value=self.char_pad_token_id, dtype=torch.long
        )
        for i, sentence_char_ids_list in enumerate(source_char_ids_list_of_lists):
            num_subwords_in_sentence = len(sentence_char_ids_list)
            for j, char_tensor_for_subword in enumerate(sentence_char_ids_list):
                padded_source_char_ids[i, j,
                :char_tensor_for_subword.size(0)] = char_tensor_for_subword  # Handle potential truncation

        collated_batch = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "source_char_ids": padded_source_char_ids,
        }
        if padded_labels is not None:
            collated_batch["labels"] = padded_labels

        # Pass through teacher-related raw data
        collated_batch["teacher_input_text"] = [item["teacher_input_text"] for item in batch]
        collated_batch["teacher_src_nllb_code"] = [item["teacher_src_nllb_code"] for item in batch]
        # If target text was also stored for teacher (e.g. item["teacher_target_text"])
        # collated_batch["teacher_target_text"] = [item["teacher_target_text"] for item in batch]
        # collated_batch["teacher_tgt_nllb_code"] = [item["teacher_tgt_nllb_code"] for item in batch]

        return collated_batch