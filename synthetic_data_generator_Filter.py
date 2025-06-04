import evaluate
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from typing import List, Dict, Optional
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_SUBWORD_SEQ_LEN = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SyntheticDataGeneratorAndFilter:
    def __init__(self,
                 teacher_model_name_or_path: str,
                 teacher_tokenizer_name_or_path: str,  # Can be same as model
                 sbert_model_instance: Optional[SentenceTransformer] = None,
                 src_lang_nllb_code: str,  # e.g., "luo_Latn"
                 tgt_lang_nllb_code: str,  # e.g., "swa_Latn"
                 device: str = DEVICE):
        self.device = device
        self.src_lang_nllb = src_lang_nllb_code
        self.tgt_lang_nllb = tgt_lang_nllb_code

        logger.info(f"Loading teacher model ({teacher_model_name_or_path}) for synthetic generation and filtering.")
        # For generation, can use BNB if model is large
        bnb_config_teacher = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        # If TEACHER_CKPT is very large (like NLLB 3.3B), loading it fully multiple times is an issue.
        # The pipeline will load it once. For other filters, we might need to be careful.
        # Let's assume for now pipeline handles its own model loading.
        # For RTT and LM-fluency, we might need separate instances or a shared one.
        # Using a single instance for all teacher model needs to save memory.
        self.teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
            teacher_model_name_or_path,
            quantization_config=bnb_config_teacher,  # Use for large teacher
            device_map="auto"
        ).eval()  # Set to eval mode
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_tokenizer_name_or_path,
            src_lang=self.src_lang_nllb  # For NLLB, set src_lang
        )
        logger.info("Teacher model and tokenizer loaded.")

        self.sbert = sbert_model_instance
        logger.info("SBERT model loaded.")
        self.chrf_metric = evaluate.load('chrf')

    def generate_synthetic_data(self, src_sentences: List[str], batch_size: int = 32) -> List[Dict[str, str]]:
        """Generates synthetic target sentences from source sentences."""
        logger.info(
            f"Generating synthetic data from {len(src_sentences)} source sentences ({self.src_lang_nllb} -> {self.tgt_lang_nllb}).")
        # Device for pipeline: 0 for first GPU if cuda, -1 for CPU
        pipeline_device = 0 if self.device == 'cuda' else -1

        # Use the already loaded teacher model and tokenizer for the pipeline
        translation_pipeline = pipeline(
            "translation",
            model=self.teacher_model,
            tokenizer=self.teacher_tokenizer,  # Pass tokenizer with src_lang already set
            # src_lang=self.src_lang_nllb, # Not needed if tokenizer is pre-configured
            tgt_lang=self.tgt_lang_nllb,  # NLLB pipeline needs forced_bos_token_id, not tgt_lang here.
            # We will set it during generation call.
            device=pipeline_device,
            batch_size=batch_size // 2 if pipeline_device == 0 else batch_size
            # Smaller batch for pipeline on GPU if needed
        )
        tgt_lang_id = self.teacher_tokenizer.lang_code_to_id[self.tgt_lang_nllb]

        synthetic_pairs = []
        # Process in batches using the pipeline
        for i in tqdm(range(0, len(src_sentences), batch_size), desc="Synthetic Generation"):
            batch_src = src_sentences[i:i + batch_size]
            # For NLLB pipeline, to set target language, we pass forced_bos_token_id to generate_kwargs
            generated_batch = translation_pipeline(batch_src, generate_kwargs={"forced_bos_token_id": tgt_lang_id})
            for original_src, translation_output in zip(batch_src, generated_batch):
                synthetic_pairs.append({
                    "original_src": original_src,  # Dholuo
                    "synthetic_tgt": translation_output['translation_text']  # Swahili
                })
        return synthetic_pairs

    def filter_round_trip(self, synthetic_pairs_ds: HFDataset, threshold: float = 0.45,
                          batch_size: int = 16) -> HFDataset:
        logger.info(f"Applying Round Trip Translation (RTT) filter (threshold: chrF >= {threshold}).")
        # RTT: synthetic_tgt (e.g. Swahili) back to original_src lang (e.g. Dholuo)
        pipeline_device = 0 if self.device == 'cuda' else -1
        # Create a new pipeline for back-translation
        # Need to set src_lang on tokenizer for NLLB
        back_translation_tokenizer = AutoTokenizer.from_pretrained(
            self.teacher_tokenizer.name_or_path, src_lang=self.tgt_lang_nllb
            # Source for back-translation is target of forward
        )
        rtt_pipeline = pipeline(
            "translation",
            model=self.teacher_model,  # Can reuse the loaded teacher model
            tokenizer=back_translation_tokenizer,
            # src_lang=self.tgt_lang_nllb, # Already set in tokenizer
            # tgt_lang=self.src_lang_nllb, # Will be set by forced_bos_token_id
            device=pipeline_device,
            batch_size=batch_size
        )
        src_lang_id_for_rtt = self.teacher_tokenizer.lang_code_to_id[self.src_lang_nllb]

        original_src_sents = synthetic_pairs_ds["original_src"]
        synthetic_tgt_sents = synthetic_pairs_ds["synthetic_tgt"]
        back_translated_src_sents = []

        for i in tqdm(range(0, len(synthetic_tgt_sents), batch_size), desc="RTT Back-translation"):
            batch_to_translate = synthetic_tgt_sents[i:i + batch_size]
            translated_batch = rtt_pipeline(batch_to_translate,
                                            generate_kwargs={"forced_bos_token_id": src_lang_id_for_rtt})
            back_translated_src_sents.extend([o['translation_text'] for o in translated_batch])

        scores = []
        for orig_src, back_trans_src in zip(original_src_sents, back_translated_src_sents):
            # chrF returns score 0-1, threshold might be 0.4. Original code had threshold*100.
            score = self.chrf_metric.compute(predictions=[back_trans_src], references=[[orig_src]])['score']
            scores.append(score)

        indices_to_keep = [i for i, score in enumerate(scores) if score >= threshold]
        logger.info(f"RTT filter: {len(indices_to_keep)} / {len(synthetic_pairs_ds)} pairs kept.")
        return synthetic_pairs_ds.select(indices_to_keep)

    def filter_semantic_similarity(self, dataset: HFDataset, threshold: float = 0.75,
                                   batch_size: int = 64) -> HFDataset:
        logger.info(f"Applying Semantic Similarity filter (SBERT cosine sim >= {threshold}).")
        src_sents = dataset["original_src"]
        tgt_sents = dataset["synthetic_tgt"]

        # SBERT expects sentences, ensure they are not empty
        valid_indices = [i for i, (s, t) in enumerate(zip(src_sents, tgt_sents)) if s and t]
        if len(valid_indices) < len(src_sents):
            logger.warning(
                f"Found {len(src_sents) - len(valid_indices)} empty sentence(s) in input to semantic filter. Processing only valid ones.")
            dataset = dataset.select(valid_indices)
            src_sents = dataset["original_src"]
            tgt_sents = dataset["synthetic_tgt"]

        if not src_sents:  # If all were empty or became empty
            logger.warning("No valid sentences left for semantic similarity filtering.")
            return dataset

        emb_src = self.sbert.encode(src_sents, convert_to_tensor=True, device=self.device, batch_size=batch_size)
        emb_tgt = self.sbert.encode(tgt_sents, convert_to_tensor=True, device=self.device, batch_size=batch_size)
        similarities = F.cosine_similarity(emb_src, emb_tgt).cpu().tolist()

        indices_to_keep = [i for i, sim in enumerate(similarities) if sim >= threshold]
        logger.info(f"Semantic filter: {len(indices_to_keep)} / {len(dataset)} pairs kept.")
        return dataset.select(indices_to_keep)

    def filter_lm_fluency(self, dataset: HFDataset, lang_to_check_key: str, threshold: float = -3.5,
                          batch_size: int = 16) -> HFDataset:
        """Filters based on LM perplexity (negative log loss) of sentences in the specified language key."""
        logger.info(f"Applying LM Fluency filter on '{lang_to_check_key}' (threshold: neg_loss >= {threshold}).")
        sentences_to_check = dataset[lang_to_check_key]
        lm_scores = []

        # Determine language for tokenizer, default to target if checking synthetic_tgt
        lang_code_for_lm = self.tgt_lang_nllb if lang_to_check_key == "synthetic_tgt" else self.src_lang_nllb

        # Use a tokenizer appropriate for the language being checked for fluency
        lm_tokenizer = AutoTokenizer.from_pretrained(self.teacher_tokenizer.name_or_path, src_lang=lang_code_for_lm)

        for i in tqdm(range(0, len(sentences_to_check), batch_size), desc=f"LM Fluency on {lang_to_check_key}"):
            batch_sents = sentences_to_check[i:i + batch_size]
            if not any(batch_sents):  # Skip if batch is all empty strings
                lm_scores.extend([-float('inf')] * len(batch_sents))  # Assign a very low score
                continue

            inputs = lm_tokenizer(batch_sents, return_tensors="pt", padding=True, truncation=True,
                                  max_length=MAX_SUBWORD_SEQ_LEN).to(self.device)
            input_ids = inputs.input_ids
            # For perplexity, labels are the same as input_ids
            labels = input_ids.clone()
            # In many Causal LM setups, pad token labels are set to -100. For Seq2Seq eval as LM, this might also be done.
            labels[labels == lm_tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = self.teacher_model(**inputs, labels=labels)  # teacher_model acts as LM
                # Loss is already averaged; for sentence-level, we might want to sum and normalize by non-pad tokens
                # However, the loss returned by HF is usually suitable for comparison.
                # For multiple sentences in batch, the loss is the mean. We need per-sentence.
                # This is tricky with HF's batch loss. A simple way is to use the batch loss for all items in batch,
                # or iterate one by one (slow).
                # For now, let's approximate with batch loss; better would be per-sentence perplexity.
                # A common proxy: use -loss. Higher is better (less perplexity).
                neg_loss = -outputs.loss.item()
            lm_scores.extend([neg_loss] * len(batch_sents))

        indices_to_keep = [i for i, score in enumerate(lm_scores) if score >= threshold]
        logger.info(f"LM Fluency filter on '{lang_to_check_key}': {len(indices_to_keep)} / {len(dataset)} pairs kept.")
        return dataset.select(indices_to_keep)

    def apply_all_filters(self, synthetic_pairs_ds: HFDataset,
                          rtt_threshold: float = 0.45,
                          semantic_threshold: float = 0.75,
                          lm_fluency_threshold_tgt: float = -3.5
                          ) -> HFDataset:
        logger.info(f"Applying all filters to synthetic dataset of size {len(synthetic_pairs_ds)}...")
        filtered_ds = self.filter_round_trip(synthetic_pairs_ds, threshold=rtt_threshold)
        if not len(filtered_ds): logger.warning("Dataset empty after RTT filter."); return filtered_ds
        filtered_ds = self.filter_semantic_similarity(filtered_ds, threshold=semantic_threshold)
        if not len(filtered_ds): logger.warning("Dataset empty after semantic filter."); return filtered_ds
        filtered_ds = self.filter_lm_fluency(filtered_ds, lang_to_check_key="synthetic_tgt",
                                             threshold=lm_fluency_threshold_tgt)
        if not len(filtered_ds): logger.warning("Dataset empty after LM fluency filter."); return filtered_ds
        logger.info(f"Finished all filters. Final synthetic dataset size: {len(filtered_ds)}")
        return filtered_ds