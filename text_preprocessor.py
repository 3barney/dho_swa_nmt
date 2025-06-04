import re
from typing import List
import logging

from sympy import false

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SENTENCEX_AVAILABLE=false

try:
    from sentencex import segment as sentencex_segment

    SENTENCEX_AVAILABLE = True
    print("Using sentencex for sentence segmentation.")
except ImportError:
    print("WARNING: sentencex library not found. Falling back to NLTK for sentence segmentation.")
    print("This is less ideal for non-English languages. Please install sentencex: pip install sentencex")
    try:
        import nltk

        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize

        SENTENCEX_AVAILABLE = False  # Mark as false even if NLTK is found, to indicate it's a fallback
    except ImportError:
        print("ERROR: NLTK not found either. Sentence segmentation will not work.")
        print("Please install NLTK: pip install nltk")
        SENTENCEX_AVAILABLE = False  # Explicitly false


        # Define a dummy sent_tokenize if NLTK is also missing to prevent further crashes,
        # but this will be very poor for actual segmentation.
        def sent_tokenize(text, language='english'):
            print(
                f"CRITICAL WARNING: No sentence tokenizer available. Splitting by newline as a last resort for text: '{text[:100]}...'")
            return text.splitlines()




class TextPreprocessor:
    def __init__(self, src_lang_iso: str = 'en', tgt_lang_iso: str = 'en'):
        self.src_lang_iso = src_lang_iso
        self.tgt_lang_iso = tgt_lang_iso
        logger.info(
            f"TextPreprocessor initialized for src: {src_lang_iso}, tgt: {tgt_lang_iso}. Using sentencex: {SENTENCEX_AVAILABLE}")

    def _remove_boilerplate(self, text: str) -> str:
        lines = text.splitlines()
        keep = []
        boilerplate_patterns = [
            r'^(evaluation only|created with|copyright|©|\(c\))', r'https?://\S+', r'^\s*$',
            # Add Dholuo/Swahili specific patterns if known, e.g. common website footers
        ]
        combined_boilerplate_regex = re.compile('|'.join(boilerplate_patterns), re.IGNORECASE)
        for l_idx, l in enumerate(lines):
            s = l.strip()
            if combined_boilerplate_regex.search(s): continue
            if 0 < len(s) < 50 and s.isupper() and sum(c.isalpha() for c in s) > 0.7 * len(s):
                if not (l_idx > 0 and len(lines[l_idx - 1].strip()) > 0 and \
                        l_idx < len(lines) - 1 and len(lines[l_idx + 1].strip()) > 0):
                    continue
            keep.append(l)
        return '\n'.join(keep)

    def _normalize_whitespace_and_punctuation(self, text: str) -> str:
        text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\xa0', ' ')  # Non-breaking space
        text = re.sub(r'\s+', ' ', text).strip()
        # Ensure space around standardized ellipses and after common punctuation
        text = text.replace('...', ' … ')
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)  # Add space after
        text = re.sub(r'\s*([\(])\s*', r' \1', text)  # Space before opening parens
        text = re.sub(r'\s*([\)])\s*', r'\1 ', text)  # Space after closing parens
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_text(self, text: str) -> str:
        text = self._remove_boilerplate(text)
        text = self._normalize_whitespace_and_punctuation(text)
        return text.lower()  # Lowercase at the end of basic cleaning

    def segment_sentences(self, text: str, lang_code: str) -> List[str]:
        # lang_code here should be short ISO code like 'sw', 'luo', 'en'
        if SENTENCEX_AVAILABLE:
            try:
                return list(sentencex_segment(lang_code, text))
            except Exception as e:
                logger.warning(
                    f"sentencex segmentation failed for lang '{lang_code}': {e}. Falling back to NLTK with 'english' rules.")
                return nltk.sent_tokenize(text, language='english')  # Fallback
        elif 'nltk' in globals() and 'sent_tokenize' in globals():
            return nltk.sent_tokenize(text, language='english')  # Fallback
        else:  # Absolute last resort
            return [s.strip() for s in text.split('.') if s.strip()]

    def linguistic_process_swahili(self, text: str) -> str:
        # Specific Swahili morphological normalization can be added here if proven beneficial and doesn't conflict with subword/char tokenization.
        return text

    def linguistic_process_dholuo(self, text: str) -> str:
        # Diacritic removal for Dholuo is generally NOT recommended as it changes meaning.
        return text
