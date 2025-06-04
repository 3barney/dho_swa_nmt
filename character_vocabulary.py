import logging
from collections import Counter
from typing import List, Dict, Optional
from transformers import AutoTokenizer
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHAR_PAD_TOKEN = "<CPAD>"  # Character PAD
CHAR_UNK_TOKEN = "<CUNK>"  # Character UNK
CHAR_SOW_TOKEN = "<SOW>"  # Start of Word/Subword
CHAR_EOW_TOKEN = "<EOW>"  # End of Word/Subword

class CharacterVocabulary:
    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens if special_tokens else [CHAR_PAD_TOKEN, CHAR_UNK_TOKEN, CHAR_SOW_TOKEN,
                                                                     CHAR_EOW_TOKEN]
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self._build_initial_vocab()

    def _build_initial_vocab(self):
        for token in self.special_tokens:
            if token not in self.char2id:
                new_id = len(self.char2id)
                self.char2id[token] = new_id
                self.id2char[new_id] = token
        self.pad_id = self.char2id[CHAR_PAD_TOKEN]
        self.unk_id = self.char2id[CHAR_UNK_TOKEN]
        self.sow_id = self.char2id[CHAR_SOW_TOKEN]
        self.eow_id = self.char2id[CHAR_EOW_TOKEN]

    def build_from_subword_tokenizer(self, subword_tokenizer: AutoTokenizer, sentences: List[str]):
        """
        Builds character vocabulary from subword tokens, including characters from special subword tokens.
        """
        char_counts = Counter()
        logger.info("Building character vocabulary from subword tokenizer and sentences...")

        # Add characters from the subword tokenizer's own vocabulary
        if hasattr(subword_tokenizer, 'get_vocab'):
            subword_vocab = subword_tokenizer.get_vocab()
            for subword_token_str in tqdm(subword_vocab.keys(), desc="Processing subword vocab chars"):
                char_counts.update(subword_token_str)
        else:  # Fallback for tokenizers without get_vocab, less comprehensive
            for special_token in subword_tokenizer.all_special_tokens:
                char_counts.update(special_token)
            logger.warning(
                "Subword tokenizer does not have `get_vocab` method. Character vocab for subwords might be incomplete.")

        # Add characters from the actual sentences after subword tokenization
        for sentence in tqdm(sentences, desc="Processing sentence chars"):
            # Tokenize to subwords, then get characters from those subwords
            # Using tokenizer.tokenize() gives the string representation of subwords directly
            subword_tokens = subword_tokenizer.tokenize(sentence)
            for subword_token_str in subword_tokens:
                char_counts.update(subword_token_str)

        for char, _ in char_counts.most_common():
            if char not in self.char2id:
                new_id = len(self.char2id)
                self.char2id[char] = new_id
                self.id2char[new_id] = char
        logger.info(f"Built character vocabulary with {len(self.char2id)} unique characters.")

    def encode_subword(self, subword_token_str: str, max_char_len: int) -> List[int]:
        """Encodes a single subword string to a list of char IDs, with SOW, EOW, padding and truncation."""
        char_ids = [self.sow_id]
        char_ids.extend(self.char2id.get(char, self.unk_id) for char in subword_token_str)
        char_ids.append(self.eow_id)

        if len(char_ids) > max_char_len:
            char_ids = char_ids[:max_char_len - 1] + [self.eow_id]  # Ensure EOW is last if truncated
        else:
            char_ids.extend([self.pad_id] * (max_char_len - len(char_ids)))
        return char_ids

    def build_from_data(self,
                        all_sentences_for_vocab: List[str],
                        subword_tokenizer: AutoTokenizer,
                        task_prefixes: Optional[List[str]] = None):
        """
        Builds or extends the character vocabulary from various data sources.

        This method collects all unique characters from:
        1. The vocabulary of the provided `subword_tokenizer`.
        2. All special tokens defined in the `subword_tokenizer`.
        3. Optional `task_prefixes`.
        4. All `all_sentences_for_vocab` after they are tokenized into subwords.

        Characters already present in the vocabulary (e.g., initial special tokens)
        are not added again.

        Args:
            all_sentences_for_vocab (List[str]): A list of sentences (strings) to
                                                 extract characters from.
            subword_tokenizer (AutoTokenizer): A Hugging Face tokenizer instance used
                                               to process sentences into subwords and
                                               to source characters from its vocabulary.
            task_prefixes (Optional[List[str]]): An optional list of task-specific prefix
                                                 strings to also include characters from.
        """
        logger.info("Starting to build character vocabulary from data...")
        char_counts = Counter()

        # 1. Add characters from the subword tokenizer's own vocabulary
        if hasattr(subword_tokenizer, 'get_vocab') and callable(subword_tokenizer.get_vocab):
            logger.info("Processing characters from subword tokenizer's vocabulary...")
            subword_vocab_dict = subword_tokenizer.get_vocab()
            for subword_token_str in tqdm(subword_vocab_dict.keys(), desc="Subword Vocab Chars"):
                char_counts.update(subword_token_str)
        else:
            logger.warning("Subword tokenizer does not have a callable `get_vocab` method. "
                           "Characters from its vocabulary won't be explicitly added. "
                           "Relying on its special tokens and provided sentences.")

        # 2. Add characters from the subword tokenizer's special tokens
        logger.info("Processing characters from subword tokenizer's special tokens...")
        if hasattr(subword_tokenizer, 'all_special_tokens') and subword_tokenizer.all_special_tokens:
            for special_token in tqdm(subword_tokenizer.all_special_tokens, desc="Subword Special Tokens Chars"):
                char_counts.update(special_token)
        else:
            logger.warning("Subword tokenizer does not have `all_special_tokens` or it's empty.")

        # 3. Add characters from task prefixes, if provided
        if task_prefixes:
            logger.info("Processing characters from task prefixes...")
            for prefix in tqdm(task_prefixes, desc="Task Prefix Chars"):
                char_counts.update(prefix)

        # 4. Process characters from the provided sentences
        logger.info("Processing characters from provided sentences...")
        for sentence in tqdm(all_sentences_for_vocab, desc="Sentence Chars"):
            if not sentence:  # Skip empty sentences
                continue
            # Tokenize the sentence into subword strings
            subword_strings = subword_tokenizer.tokenize(sentence)
            for subword_str in subword_strings:
                char_counts.update(subword_str)

        # Add new characters to the vocabulary
        logger.info("Adding new characters to the vocabulary...")
        added_chars_count = 0
        for char, count in tqdm(char_counts.most_common(), desc="Finalizing Char Vocab"):
            if char not in self.char2id:  # Add only if new
                new_id = len(self.char2id)
                self.char2id[char] = new_id
                self.id2char[new_id] = char
                added_chars_count += 1

        logger.info(f"Added {added_chars_count} new unique characters to the vocabulary.")
        logger.info(f"Total character vocabulary size: {len(self.char2id)} unique characters.")

        # Verification: Ensure special char tokens are still correctly mapped (they should be)
        for token_name, token_val in [("CHAR_PAD_TOKEN", CHAR_PAD_TOKEN),
                                      ("CHAR_UNK_TOKEN", CHAR_UNK_TOKEN),
                                      ("CHAR_SOW_TOKEN", CHAR_SOW_TOKEN),
                                      ("CHAR_EOW_TOKEN", CHAR_EOW_TOKEN)]:
            if token_val not in self.char2id:
                logger.error(
                    f"CRITICAL ERROR: Special token {token_name} ('{token_val}') is missing from char2id after build_from_data.")
            elif self.id2char.get(self.char2id[token_val]) != token_val:
                logger.error(f"CRITICAL ERROR: Special token {token_name} ('{token_val}') has inconsistent mapping.")

    def __len__(self):
        return len(self.char2id)

    def get_pad_id(self) -> int:
        return self.pad_id

    def get_unk_id(self) -> int:
        return self.unk_id

    def get_sow_id(self) -> int:
        return self.sow_id

    def get_eow_id(self) -> int:
        return self.eow_id