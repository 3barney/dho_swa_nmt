import logging

from pathlib import Path
from text_preprocessor import TextPreprocessor
from tqdm.auto import tqdm
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SRC_LANG_SHORT = "luo"
TGT_LANG_SHORT = "swa"

class FileProcessor:
    """Processes raw files into lists of sentences or pairs."""

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor

    def load_and_process_monolingual_dir(self, folder_path: Path, lang_iso_for_segmentation: str,
                                         lang_for_linguistic_processing: str) -> List[str]:
        all_sentences = []
        if not folder_path.is_dir():
            logger.warning(f"Monolingual directory not found: {folder_path}")
            return all_sentences
        for file_path in tqdm(list(folder_path.glob('*.txt')), desc=f"Processing mono {lang_iso_for_segmentation}"):
            try:
                raw_text = file_path.read_text(encoding='utf-8')
                cleaned_text = self.preprocessor.clean_text(raw_text)  # Basic cleaning & lowercasing
                sentences = self.preprocessor.segment_sentences(cleaned_text, lang_code=lang_iso_for_segmentation)

                processed_sentences = []
                for s in sentences:
                    if s.strip():
                        if lang_for_linguistic_processing == SRC_LANG_SHORT:  # Dholuo
                            processed_s = self.preprocessor.linguistic_process_dholuo(s)
                        elif lang_for_linguistic_processing == TGT_LANG_SHORT:  # Swahili
                            processed_s = self.preprocessor.linguistic_process_swahili(s)
                        else:
                            processed_s = s  # Already lowercased
                        if processed_s:  # Ensure not empty after linguistic processing
                            processed_sentences.append(processed_s)
                all_sentences.extend(processed_sentences)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        # Remove exact duplicates, preserve order for potential reproducibility issues if set() is used.
        unique_sentences = []
        seen = set()
        for sentence in all_sentences:
            if sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        return unique_sentences