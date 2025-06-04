import logging

from pathlib import Path
from text_preprocessor import TextPreprocessor
from tqdm.auto import tqdm
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SRC_LANG_SHORT = "luo"
TGT_LANG_SHORT = "swa"


class FileProcessor:
    """Processes raw files into lists of sentences or pairs."""

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor

    def load_and_process_monolingual_dir(self,
                                         folder_path: Path,
                                         lang_short_code_for_segmentation: str,  # "luo", "swa"
                                         ) -> List[str]:
        all_sentences = []
        if not folder_path.exists() or not folder_path.is_dir():
            logger.warning(f"Monolingual directory not found or not a directory: {folder_path}")
            return all_sentences

        file_paths = list(folder_path.glob('*.txt'))
        if not file_paths:
            logger.warning(f"No .txt files found in monolingual directory: {folder_path}")
            return all_sentences

        for file_path in tqdm(file_paths,
                              desc=f"Processing mono '{lang_short_code_for_segmentation}' in {folder_path.name}"):
            try:
                raw_text = file_path.read_text(encoding='utf-8', errors='ignore')
                if not raw_text.strip():
                    continue

                cleaned_text = self.preprocessor.clean_text(raw_text)
                sentences = self.preprocessor.segment_sentences(cleaned_text,
                                                                lang_code=lang_short_code_for_segmentation)

                processed_sentences = []
                for s in sentences:
                    s_stripped = s.strip()
                    if s_stripped:
                        # Apply linguistic processing based on the language being processed
                        if lang_short_code_for_segmentation == SRC_LANG_SHORT:  # Dholuo original constant
                            processed_s = self.preprocessor.linguistic_process_dholuo(s_stripped)
                        elif lang_short_code_for_segmentation == TGT_LANG_SHORT:  # Swahili original constant
                            processed_s = self.preprocessor.linguistic_process_swahili(s_stripped)
                        else:
                            # Fallback if language doesn't match primary SRC/TGT or example languages
                            logger.warning(
                                f"Linguistic processing not specifically defined for {lang_short_code_for_segmentation}, using cleaned text.")
                            processed_s = s_stripped

                        if processed_s:
                            processed_sentences.append(processed_s)
                all_sentences.extend(processed_sentences)
            except Exception as e:
                logger.error(f"Error processing monolingual file {file_path}: {e}", exc_info=True)

        unique_sentences = []
        seen = set()
        for sentence in all_sentences:
            if sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        logger.info(
            f"Loaded {len(unique_sentences)} unique sentences from {folder_path.name} for lang '{lang_short_code_for_segmentation}'.")
        return unique_sentences

    def load_and_process_parallel_files(self,
                                        # These are the actual language content identifiers
                                        content_lang1_file: Path,
                                        content_lang2_file: Path,
                                        content_lang1_short_code: str,  # e.g. "luo"
                                        content_lang2_short_code: str,  # e.g. "swa"
                                        ) -> List[Dict[str, str]]:  # Returns {'lang1': ..., 'lang2': ...}
        pairs = []
        if not content_lang1_file.exists():
            logger.warning(f"Content lang1 parallel file not found: {content_lang1_file}")
            return pairs
        if not content_lang2_file.exists():
            logger.warning(f"Content lang2 parallel file not found: {content_lang2_file}")
            return pairs

        try:
            with open(content_lang1_file, 'r', encoding='utf-8', errors='ignore') as f_l1, \
                    open(content_lang2_file, 'r', encoding='utf-8', errors='ignore') as f_l2:

                desc = f"Processing parallel '{content_lang1_short_code}' & '{content_lang2_short_code}'"
                for line_num, (l1_line, l2_line) in enumerate(tqdm(zip(f_l1, f_l2), desc=desc)):
                    raw_l1_sent = l1_line.strip()
                    raw_l2_sent = l2_line.strip()

                    if not raw_l1_sent or not raw_l2_sent:
                        continue

                    cleaned_l1 = self.preprocessor.clean_text(raw_l1_sent)
                    cleaned_l2 = self.preprocessor.clean_text(raw_l2_sent)

                    # Apply linguistic processing based on the actual language content
                    final_l1, final_l2 = cleaned_l1, cleaned_l2  # Default to cleaned text
                    if content_lang1_short_code == SRC_LANG_SHORT:  # Dholuo
                        final_l1 = self.preprocessor.linguistic_process_dholuo(cleaned_l1)
                    elif content_lang1_short_code == TGT_LANG_SHORT:  # Swahili
                        final_l1 = self.preprocessor.linguistic_process_swahili(cleaned_l1)

                    if content_lang2_short_code == TGT_LANG_SHORT:  # Swahili
                        final_l2 = self.preprocessor.linguistic_process_swahili(cleaned_l2)
                    elif content_lang2_short_code == SRC_LANG_SHORT:  # Dholuo
                        final_l2 = self.preprocessor.linguistic_process_dholuo(cleaned_l2)

                    if final_l1 and final_l2:
                        # Store with actual language keys first
                        pairs.append({
                            content_lang1_short_code: final_l1,
                            content_lang2_short_code: final_l2
                        })
        except Exception as e:
            logger.error(f"Error processing parallel files {content_lang1_file}, {content_lang2_file}: {e}",
                         exc_info=True)

        logger.info(f"Loaded {len(pairs)} raw pairs from {content_lang1_file.name}/{content_lang2_file.name}.")
        return pairs