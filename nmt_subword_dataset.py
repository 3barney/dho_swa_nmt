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
                 # Expects columns like "student_input_text", "target_text", "teacher_input_text", "teacher_src_nllb", "teacher_tgt_nllb"
                 student_subword_tokenizer: AutoTokenizer,  # Tokenizer for the student model
                 teacher_subword_tokenizer: AutoTokenizer,  # Tokenizer for the teacher model (for teacher_input_ids)
                 char_vocab: 'CharacterVocabulary',
                 max_subword_seq_len: int,
                 max_char_len_per_subword_incl_special: int):
        self.hf_dataset = hf_dataset
        self.student_subword_tokenizer = student_subword_tokenizer
        self.teacher_subword_tokenizer = teacher_subword_tokenizer
        self.char_vocab = char_vocab
        self.max_subword_seq_len = max_subword_seq_len
        self.max_char_len_per_subword = max_char_len_per_subword_incl_special

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.hf_dataset[idx]

        student_input_text = example["student_input_text"]
        target_text = example["target_text"]
        teacher_input_text = example["teacher_input_text"]
        teacher_src_nllb_code = example["teacher_src_nllb"]

        # --- Student Input Processing ---
        student_tokenized_output = self.student_subword_tokenizer(
            student_input_text,  # Text already has prefix
            max_length=self.max_subword_seq_len,
            truncation=True,
            padding=False
        )
        student_input_ids_list = student_tokenized_output['input_ids']
        student_attention_mask_list = student_tokenized_output['attention_mask']
        student_subword_strings = self.student_subword_tokenizer.convert_ids_to_tokens(student_input_ids_list,
                                                                                       skip_special_tokens=False)

        student_char_ids_for_each_subword_list = []
        for subword_str in student_subword_strings:
            char_ids = self.char_vocab.encode_subword_string(subword_str, self.max_char_len_per_subword)
            student_char_ids_for_each_subword_list.append(torch.tensor(char_ids, dtype=torch.long))

        item = {
            "input_ids": torch.tensor(student_input_ids_list, dtype=torch.long),  # For student model (prefixed)
            "attention_mask": torch.tensor(student_attention_mask_list, dtype=torch.long),
            "source_char_ids": student_char_ids_for_each_subword_list,
        }

        # --- Target (Labels) Processing for Student ---
        if target_text:
            # Tokenizer for student (e.g. mT5)
            # For NLLB student, tokenizer needs to be in target_lang mode
            if "nllb" in self.student_subword_tokenizer.name_or_path.lower():
                # Temporarily set teacher_tgt_nllb as the target for student NLLB tokenizer
                # This assumes student and teacher share language codes if student is NLLB
                # current_student_tgt_nllb = example.get("teacher_tgt_nllb", TGT_LANG_NLLB)  # Fallback
                with self.student_subword_tokenizer.as_target_tokenizer():  # NLLB style
                    # NLLB tokenizer uses src_lang for input, target_lang for output.
                    # It needs to know what language `target_text` is in to prepend correct code if needed.
                    # This is tricky. Usually, labels don't get language codes prepended by tokenizer,
                    # model handles that based on forced_bos_token_id or config.
                    # For `text_target` in NLLB, it expects tokenizer's `tgt_lang` to be set.
                    # For safety, let's re-init a tokenizer instance for target or ensure it's configured.
                    # This part is complex for NLLB student.
                    # Simpler: For mT5, `text_target` is fine. For NLLB, tokenizer(text_target=...) expects tokenizer.tgt_lang.
                    temp_tgt_tokenizer = AutoTokenizer.from_pretrained(self.student_subword_tokenizer.name_or_path,
                                                                       src_lang=teacher_src_nllb_code,
                                                                       tgt_lang=example.get("teacher_tgt_nllb"))

                    tgt_tokenized_output = temp_tgt_tokenizer(
                        text_target=target_text,
                        max_length=self.max_subword_seq_len, truncation=True, padding=False
                    )
            else:  # For mT5 and similar
                tgt_tokenized_output = self.student_subword_tokenizer(
                    text_target=target_text,
                    max_length=self.max_subword_seq_len, truncation=True, padding=False
                )
            item["labels"] = torch.tensor(tgt_tokenized_output['input_ids'], dtype=torch.long)

        # --- Teacher Input Processing (Unprefixed) ---
        # This assumes teacher_tokenizer is NLLB tokenizer, needing src_lang
        # We store the text and codes; actual tokenization can happen in collator or trainer for teacher
        item["teacher_input_text"] = teacher_input_text
        item["teacher_src_nllb_code"] = teacher_src_nllb_code
        # item["teacher_tgt_nllb_code"] = example["teacher_tgt_nllb"] # also pass if needed for teacher labels

        return item