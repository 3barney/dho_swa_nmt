import logging
from collections import Counter
from typing import List, Dict
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