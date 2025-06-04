import torch

from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence


MAX_CHAR_LEN_PER_SUBWORD = 16

class CustomDataCollator:
    def __init__(self, subword_pad_token_id: int, char_pad_token_id: int):
        self.subword_pad_token_id = subword_pad_token_id
        self.char_pad_token_id = char_pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        source_char_ids_list_of_lists = [item["source_char_ids"] for item in batch]  # list of list of tensors

        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.subword_pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)  # 0 for mask
        padded_labels = pad_sequence(labels_list, batch_first=True,
                                     padding_value=self.subword_pad_token_id)  # often -100 for labels

        # Pad source_char_ids
        # Each element in source_char_ids_list_of_lists is a list of char_id_tensors for one sentence.
        # All char_id_tensors within one sentence already have the same char_len (MAX_CHAR_LEN_PER_SUBWORD)
        # We need to pad the number of subwords per sentence.
        max_subwords_in_batch = padded_input_ids.size(1)  # From already padded subword_ids

        # Assuming all inner char tensors have MAX_CHAR_LEN_PER_SUBWORD due to NMTCharSubwordDataset
        # If not, max_char_len_per_subword would need to be calculated here across the batch.
        # For now, we rely on the Dataset to have enforced this.
        # The collate_fn from the browse result had a more complex way assuming varying char_len too.
        # Here, MAX_CHAR_LEN_PER_SUBWORD is fixed.

        batch_size = len(batch)
        # Shape: (batch_size, max_subwords_in_batch, MAX_CHAR_LEN_PER_SUBWORD)
        padded_source_char_ids = torch.full(
            (batch_size, max_subwords_in_batch, MAX_CHAR_LEN_PER_SUBWORD),
            fill_value=self.char_pad_token_id,
            dtype=torch.long
        )

        for i, char_ids_for_sentence in enumerate(source_char_ids_list_of_lists):
            num_subwords_in_sentence = len(char_ids_for_sentence)
            for j, char_tensor_for_subword in enumerate(char_ids_for_sentence):
                padded_source_char_ids[i, j,
                :] = char_tensor_for_subword  # Assumes char_tensor_for_subword is already correct len

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "source_char_ids": padded_source_char_ids,
            "labels": padded_labels,
        }