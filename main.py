import gc
import os
import re
import subprocess
import numpy as np
import torch
import logging
import evaluate
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm.auto import tqdm
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    # DataCollatorForSeq2Seq, # We will use our custom collate_fn
    pipeline,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from sentence_transformers import SentenceTransformer

from character_aware_model import CharacterAwareMTModel
from character_vocabulary import CharacterVocabulary
from data_collator import CustomDataCollator
from distillation_trainer import DistillationTrainer
from file_processor import FileProcessor
from nmt_subword_dataset import NMTCharSubwordDataset
from synthetic_data_generator_Filter import SyntheticDataGeneratorAndFilter
from text_preprocessor import TextPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------
# CONFIGURATION
# -----------------------
PROJECT_NAME = "EFFICIENT LOW RESOURCE DHOLUO–SWAHILI CHARACTER-AWARE MT VIA MULTILINGUAL MODEL KD"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")

# NLLB Language codes
SRC_LANG_NLLB = "luo_Latn"
TGT_LANG_NLLB = "swh_Latn"

# Short language codes for file naming, sentencex, etc.
SRC_LANG_SHORT = "luo"
TGT_LANG_SHORT = "swh"

# DATA_DIRECTORY = Path("data/parallel")  # Main directory for user's data
# DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
#
# # User-provided Monolingual data paths
# USER_MONO_SRC_DIR = DATA_DIRECTORY / "mono_luo"  # Expect .txt files inside
# USER_MONO_TGT_DIR = DATA_DIRECTORY / "mono_swa"  # Expect .txt files inside
#
# # User-provided Parallel data paths (Moses format: one file for source, one for target)
# USER_PARALLEL_TRAIN_SRC_FILE = DATA_DIRECTORY / f"train.{SRC_LANG_SHORT}"
# USER_PARALLEL_TRAIN_TGT_FILE = DATA_DIRECTORY / f"train.{TGT_LANG_SHORT}"
# USER_PARALLEL_DEV_SRC_FILE = DATA_DIRECTORY / f"dev.{SRC_LANG_SHORT}"  # Optional, but highly recommended
# USER_PARALLEL_DEV_TGT_FILE = DATA_DIRECTORY / f"dev.{TGT_LANG_SHORT}"  # Optional

# BASE_PROJECT_DIR = Path(".")
BASE_PROJECT_DIR = Path("/content")
DATA_ROOT_DIR = BASE_PROJECT_DIR / "data"

USER_MONO_SRC_DIR = DATA_ROOT_DIR / "monolingual" / "collected"
# Your 'processed' folder for Swahili monolingual data (currently empty, but path defined)
USER_MONO_TGT_DIR = DATA_ROOT_DIR / "monolingual" / "processed" # Or "mono_swa" if you create it

# Parallel data path
# Your parallel data is a single CSV file in 'data/parallel/'
# The script was expecting separate .luo and .swa files (Moses format).
# You will need to adjust the `FileProcessor.load_and_process_parallel_files` method
# or pre-process your CSV into two separate files (one for Dholuo, one for Swahili).

# For now, let's define the path to your CSV.
# You'll need to specify the CSV filename. Let's assume it's 'dholuo_swahili_parallel.csv'
PARALLEL_CSV_FILE = DATA_ROOT_DIR / "parallel" / "dholuo_swahili_parallel.csv"

# The script's existing structure for parallel files is (Moses format):
# USER_PARALLEL_TRAIN_SRC_FILE = DATA_ROOT_DIR / "parallel" / f"train.{SRC_LANG_SHORT}"
# USER_PARALLEL_TRAIN_TGT_FILE = DATA_ROOT_DIR / "parallel" / f"train.{TGT_LANG_SHORT}"
# USER_PARALLEL_DEV_SRC_FILE = DATA_ROOT_DIR / "parallel" / f"dev.{SRC_LANG_SHORT}"
# USER_PARALLEL_DEV_TGT_FILE = DATA_ROOT_DIR / "parallel" / f"dev.{TGT_LANG_SHORT}"

# You have two main options for the parallel data:
# Option A: Preprocess your CSV into the two separate files expected by the script.
#           (e.g., train.luo and train.swa) and place them in DATA_ROOT_DIR / "parallel"
# Option B: Modify the `FileProcessor.load_and_process_parallel_files` method in your script
#           to read directly from the CSV.

# Let's assume for now you will pre-process your CSV into two files named:
# 'train_parallel.luo' and 'train_parallel.swa' and place them in 'data/parallel/'
# If you have a separate dev split from the CSV, name them 'dev_parallel.luo' and 'dev_parallel.swa'

USER_PARALLEL_TRAIN_SRC_FILE = DATA_ROOT_DIR / "parallel" / f"train_parallel.{SRC_LANG_SHORT}"
USER_PARALLEL_TRAIN_TGT_FILE = DATA_ROOT_DIR / "parallel" / f"train_parallel.{TGT_LANG_SHORT}"
# If you create dev files from your CSV:
USER_PARALLEL_DEV_SRC_FILE = DATA_ROOT_DIR / "parallel" / f"dev_parallel.{SRC_LANG_SHORT}"
USER_PARALLEL_DEV_TGT_FILE = DATA_ROOT_DIR / "parallel" / f"dev_parallel.{TGT_LANG_SHORT}"


# Ensure directories exist (optional, as the script might do this later, but good for clarity)
USER_MONO_SRC_DIR.mkdir(parents=True, exist_ok=True)
USER_MONO_TGT_DIR.mkdir(parents=True, exist_ok=True) # For Swahili monolingual, if you add some
(DATA_ROOT_DIR / "parallel").mkdir(parents=True, exist_ok=True)




# Option 2: Download example data (e.g., English-German from OpusTatoeba)
# This is primarily for testing the pipeline if luo-swa data is not immediately available.
# EXAMPLE_DATA_DIR = Path("/content/example_opus_data")
# EXAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
# EXAMPLE_SRC_LANG_OPUS = "en"
# EXAMPLE_TGT_LANG_OPUS = "de"
# EXAMPLE_CORPUS_NAME = "Tatoeba"  # Opus corpus name

# For creating dummy files if no data is provided at all (just for script to not crash immediately)
# This allows testing parts of the pipeline structure.
# CREATE_DUMMY_DATA_IF_NONE_EXIST = False

# Model checkpoints
TEACHER_CKPT = 'facebook/nllb-200-distilled-600M'
STUDENT_CKPT = 'google/mt5-small'

# Character CNN parameters
CHAR_EMB_DIM = 64  # Reduced for memory
CHAR_CNN_KERNEL_SIZES = [2, 3, 4]  # Fewer/smaller kernels
CHAR_CNN_NUM_FILTERS = 64  # Reduced number of filters

# Special character tokens
CHAR_PAD_TOKEN = "<CPAD>"  # Character PAD
CHAR_UNK_TOKEN = "<CUNK>"  # Character UNK
CHAR_SOW_TOKEN = "<SOW>"  # Start of Word/Subword
CHAR_EOW_TOKEN = "<EOW>"  # End of Word/Subword
SPECIAL_CHAR_TOKENS = [CHAR_PAD_TOKEN, CHAR_UNK_TOKEN, CHAR_SOW_TOKEN, CHAR_EOW_TOKEN]

# Training hyperparameters
MAX_SUBWORD_SEQ_LEN = 128
MAX_CHAR_LEN_PER_SUBWORD = 16  # Max characters per subword token (for padding char sequences), INCLUDES SOW/EOW
BATCH_SIZE = 8  # For Colab Free GPU. For T4, can try 16.
GRAD_ACCUM_STEPS = 4  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS (32)
LEARNING_RATE = 3e-4  # Slightly higher for smaller models / from-scratch components
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5  # Increase for real training (e.g., 10-20)
KD_ALPHA = 0.7
REP_ALPHA = 0.3
PATIENT_REP_LAYERS = [-1]  # Distill from last encoder hidden layer

# Evaluation & Logging
EVAL_STEPS = 250  # Evaluate more frequently for low-resource
LOGGING_STEPS = 50
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 3

# Metrics - initialize once
bleu_metric = evaluate.load('sacrebleu')
chrf_metric = evaluate.load('chrf')  # Returns score 0-1 by default
bertscore_metric = evaluate.load('bertscore')
comet_metric = None  # Initialize later due to potential download/resource issues
BLEURT_CKPT = 'BLEURT-20'  # General purpose checkpoint
bleurt_metric = None

try:
    comet_metric = evaluate.load('comet', config_name='eamt22-cometinho-da')  # Smaller COMET model
    logger.info(f"Successfully loaded COMET: eamt22-cometinho-da")
except Exception as e:
    logger.warning(f"Could not load preferred COMET (eamt22-cometinho-da): {e}. COMET will be skipped.")

try:
    bleurt_metric = evaluate.load('bleurt', checkpoint=BLEURT_CKPT)
    logger.info(f"Successfully loaded BLEURT: {BLEURT_CKPT}")
except Exception as e:
    logger.warning(f"Could not load BLEURT with '{BLEURT_CKPT}': {e}. BLEURT evaluation will be skipped.")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)


# -----------------------
# DATA HELPER FUNCTIONS
# -----------------------
def load_and_process_parallel_files(self, src_file_path: Path, tgt_file_path: Path,
                                    src_lang_iso_for_segmentation: str,
                                    tgt_lang_iso_for_segmentation: str,
                                    src_lang_for_linguistic_processing: str,
                                    tgt_lang_for_linguistic_processing: str
                                    ) -> List[Dict[str, str]]:
    """
    Loads and processes parallel text files (Moses format: one file per language).
    Sentences are assumed to be line-aligned between the two files.
    """
    pairs = []
    if not src_file_path.exists() or not tgt_file_path.exists():
        logger.warning(f"Parallel files not found: {src_file_path} or {tgt_file_path}")
        return pairs

    try:
        with open(src_file_path, 'r', encoding='utf-8') as f_src, \
                open(tgt_file_path, 'r', encoding='utf-8') as f_tgt:

            for line_num, (src_line, tgt_line) in enumerate(tqdm(zip(f_src, f_tgt),
                                                                 desc=f"Processing parallel {src_lang_iso_for_segmentation}-{tgt_lang_iso_for_segmentation}")):
                raw_src_sent = src_line.strip()
                raw_tgt_sent = tgt_line.strip()

                if not raw_src_sent or not raw_tgt_sent:
                    # logger.debug(f"Skipping empty line at {line_num+1} in parallel files.")
                    continue

                cleaned_src = self.preprocessor.clean_text(raw_src_sent)
                cleaned_tgt = self.preprocessor.clean_text(raw_tgt_sent)

                # Linguistic processing (currently minimal, mostly lowercasing is done by clean_text)
                if src_lang_for_linguistic_processing == SRC_LANG_SHORT:  # Dholuo
                    final_src = self.preprocessor.linguistic_process_dholuo(cleaned_src)
                else:
                    final_src = cleaned_src  # Assuming clean_text already lowercased

                if tgt_lang_for_linguistic_processing == TGT_LANG_SHORT:  # Swahili
                    final_tgt = self.preprocessor.linguistic_process_swahili(cleaned_tgt)
                else:
                    final_tgt = cleaned_tgt  # Assuming clean_text already lowercased

                if final_src and final_tgt:  # Ensure both are non-empty after all processing
                    pairs.append({
                        SRC_LANG_SHORT: final_src,
                        TGT_LANG_SHORT: final_tgt
                    })
                # else:
                # logger.debug(f"Pair became empty after processing at line {line_num+1}")
    except Exception as e:
        logger.error(f"Error processing parallel files {src_file_path}, {tgt_file_path}: {e}")
    return pairs

# -----------------------
# METRICS COMPUTATION
# -----------------------
def compute_metrics_fn(eval_preds, tokenizer: AutoTokenizer, dataset_for_comet_src: Optional[List[str]] = None):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]  # Beam search might return a tuple

    # Replace -100 in labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # SacreBLEU expects list of references for each prediction
    decoded_labels_sacrebleu = [[label] for label in decoded_labels]

    results = {}
    try:
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_sacrebleu)
        results["bleu"] = bleu_result["score"]
    except Exception as e:
        logger.error(f"Error computing BLEU: {e}")
        results["bleu"] = 0.0

    try:
        # chrF expects score 0-1. If it's 0-100, scale it in the metric config or here.
        # The evaluate.load('chrf') default is 0-100 (wordOrder=0) or 0-1 (wordOrder=2, default)
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels_sacrebleu)
        results["chrf"] = chrf_result["score"]  # Usually on a 0-100 scale or 0-1. Default is 0-1.
    except Exception as e:
        logger.error(f"Error computing chrF: {e}")
        results["chrf"] = 0.0

    try:
        # Determine language for BERTScore, default to target language short code
        # BERTScore 'lang' param uses ISO codes like 'en', 'de'.
        # For Swahili='sw', Dholuo='luo', if not directly supported,
        # 'bert-base-multilingual-cased' is often used implicitly or explicitly by setting lang="mul"
        # The original script used 'und', let's try to be more specific or use a general multilingual.
        # Check if TGT_LANG_SHORT is a known BERTScore lang. If not, 'mul' (multilingual) or None for auto-detect.
        # NLLB target lang short e.g. "swa"
        # For simplicity, if specific code not obviously supported by default models, 'mul' is safer.
        bert_lang = TGT_LANG_SHORT if TGT_LANG_SHORT in ['en', 'de', 'fr', 'es', 'zh'] else "mul"  # Heuristic
        bertscore_result = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels,
                                                    lang=bert_lang)
        results["bertscore_f1"] = np.mean(bertscore_result["f1"])
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        results["bertscore_f1"] = 0.0

    if comet_metric and dataset_for_comet_src:
        if len(dataset_for_comet_src) == len(decoded_preds) == len(decoded_labels):
            try:
                comet_result = comet_metric.compute(sources=dataset_for_comet_src, predictions=decoded_preds,
                                                    references=decoded_labels_sacrebleu)
                results["comet"] = comet_result["mean_score"]
            except Exception as e:
                logger.error(f"Error computing COMET: {e}", exc_info=True)
                results["comet"] = 0.0
        else:
            logger.warning(f"Skipping COMET: Mismatch in lengths. Sources: {len(dataset_for_comet_src)}, "
                           f"Preds: {len(decoded_preds)}, Labels: {len(decoded_labels)}")
            results["comet"] = 0.0
    else:
        if not comet_metric: logger.info("COMET metric not available, skipping.")
        if not dataset_for_comet_src: logger.info("COMET source sentences not provided, skipping.")
        results["comet"] = 0.0  # Ensure key exists

    if bleurt_metric:
        try:
            bleurt_result = bleurt_metric.compute(predictions=decoded_preds, references=decoded_labels)
            results["bleurt"] = np.mean(bleurt_result["scores"])
        except Exception as e:
            logger.error(f"Error computing BLEURT: {e}")
            results["bleurt"] = 0.0
    else:
        logger.info("BLEURT metric not available, skipping.")
        results["bleurt"] = 0.0

    # Log a few decoded examples
    for i in range(min(3, len(decoded_preds))):
        logger.info(f"Pred Ex {i}: {decoded_preds[i]}")
        logger.info(f"Label Ex {i}: {decoded_labels[i]}")
        if dataset_for_comet_src and i < len(dataset_for_comet_src):
            logger.info(f"Source Ex {i}: {dataset_for_comet_src[i]}")

    return results


# -----------------------
# MAIN PIPELINE
# -----------------------
def main():
    logger.info(PROJECT_NAME)
    logger.info(f"Starting pipeline on device: {DEVICE}")

    # --- 0. Determine active language settings ---
    global SRC_LANG_NLLB, TGT_LANG_NLLB, SRC_LANG_SHORT, TGT_LANG_SHORT
    active_src_lang_nllb = SRC_LANG_NLLB
    active_tgt_lang_nllb = TGT_LANG_NLLB
    active_src_lang_short = SRC_LANG_SHORT
    active_tgt_lang_short = TGT_LANG_SHORT

    logger.info("Using user-defined Dholuo-Swahili data paths.")
    parallel_train_src_file = USER_PARALLEL_TRAIN_SRC_FILE
    parallel_train_tgt_file = USER_PARALLEL_TRAIN_TGT_FILE
    parallel_dev_src_file = USER_PARALLEL_DEV_SRC_FILE
    parallel_dev_tgt_file = USER_PARALLEL_DEV_TGT_FILE
    mono_src_dir = USER_MONO_SRC_DIR
    mono_tgt_dir = USER_MONO_TGT_DIR



    # --- 1. Load and Preprocess Data ---
    text_preprocessor = TextPreprocessor(src_lang_short_code=active_src_lang_short,
                                         tgt_lang_short_code=active_tgt_lang_short)
    file_processor = FileProcessor(text_preprocessor)

    logger.info("Loading and processing monolingual source data...")
    mono_src_sentences = file_processor.load_and_process_monolingual_dir(
        mono_src_dir, active_src_lang_short, active_src_lang_short
    )
    logger.info(
        "Loading and processing monolingual target data...")  # For potential future use (e.g. backtranslation on target)
    mono_tgt_sentences = file_processor.load_and_process_monolingual_dir(
        mono_tgt_dir, active_tgt_lang_short, active_tgt_lang_short
    )

    logger.info("Loading and processing parallel training data...")
    parallel_train_pairs = file_processor.load_and_process_parallel_files(
        parallel_train_src_file, parallel_train_tgt_file,
        active_src_lang_short, active_tgt_lang_short,
        active_src_lang_short, active_tgt_lang_short
    )
    if not parallel_train_pairs:
        logger.error("No parallel training data loaded. Exiting.")
        return

    parallel_dev_pairs = []
    if parallel_dev_src_file and parallel_dev_tgt_file and \
            parallel_dev_src_file.exists() and parallel_dev_tgt_file.exists():
        logger.info("Loading and processing parallel dev data...")
        parallel_dev_pairs = file_processor.load_and_process_parallel_files(
            parallel_dev_src_file, parallel_dev_tgt_file,
            active_src_lang_short, active_tgt_lang_short,
            active_src_lang_short, active_tgt_lang_short
        )
    if not parallel_dev_pairs:  # If no dev set, take a small slice from train
        logger.warning("No parallel dev data found or loaded. Will attempt to split from training data.")
        if len(parallel_train_pairs) > 100:  # Ensure enough data for a split
            split_idx = min(500, int(len(parallel_train_pairs) * 0.05))  # 5% or max 500 for dev
            if split_idx == 0 and len(parallel_train_pairs) > 1: split_idx = 1  # Ensure at least 1 for dev
            parallel_dev_pairs = parallel_train_pairs[:split_idx]
            parallel_train_pairs = parallel_train_pairs[split_idx:]
            logger.info(f"Created dev set with {len(parallel_dev_pairs)} pairs from training data.")
        else:
            logger.error("Not enough training data to create a dev split. Dev set will be empty.")
            parallel_dev_pairs = []

    authentic_parallel_train_ds = HFDataset.from_list(parallel_train_pairs)
    authentic_parallel_dev_ds = HFDataset.from_list(parallel_dev_pairs) if parallel_dev_pairs else None

    # --- 2. Synthetic Data Generation & Filtering ---
    synthetic_train_ds = None
    if mono_src_sentences:
        logger.info(f"Initializing synthetic data generator for {active_src_lang_nllb} to {active_tgt_lang_nllb}...")
        synth_generator_filter = SyntheticDataGeneratorAndFilter(
            teacher_model_name_or_path=TEACHER_CKPT,
            sbert_model_name=sbert_model,
            src_lang_nllb_code=active_src_lang_nllb,
            tgt_lang_nllb_code=active_tgt_lang_nllb,
            device=DEVICE,
            use_bnb_for_teacher=True
        )

        # Convert monolingual list to HFDataset format expected by generator
        mono_src_hf_ds = HFDataset.from_dict({"original_src": mono_src_sentences})

        # Limit synthetic data generation for resource constraints if needed
        # max_synthetic_src_sents = 10000 # Example limit
        # if len(mono_src_hf_ds) > max_synthetic_src_sents:
        # mono_src_hf_ds = mono_src_hf_ds.select(range(max_synthetic_src_sents))
        # logger.info(f"Limiting synthetic data source to {max_synthetic_src_sents} sentences.")

        synthetic_pairs_raw_list = synth_generator_filter.generate_synthetic_data(
            mono_src_hf_ds["original_src"],  # Pass the list of strings
            batch_size=BATCH_SIZE * 2  # Can use larger batch for inference
        )
        if synthetic_pairs_raw_list:
            synthetic_raw_ds = HFDataset.from_list(synthetic_pairs_raw_list)
            logger.info(f"Generated {len(synthetic_raw_ds)} raw synthetic pairs.")

            synthetic_train_ds = synth_generator_filter.apply_all_filters(
                synthetic_raw_ds,
                rtt_threshold=0.40,  # chrF score for RTT
                semantic_threshold=0.65,  # SBERT cosine similarity
                lm_fluency_threshold_tgt=-4.0  # Negative log loss for fluency of synthetic target
            )
            logger.info(f"Filtered synthetic dataset size: {len(synthetic_train_ds)}")
        else:
            logger.warning("No synthetic pairs were generated.")
            synthetic_train_ds = HFDataset.from_list([])  # Empty dataset

        # Clean up large teacher model from Synth Generator if no longer needed explicitly
        del synth_generator_filter.teacher_model
        del synth_generator_filter
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    # --- 3. Combine Datasets ---
    if synthetic_train_ds and len(synthetic_train_ds) > 0:
        # Ensure columns match for concatenation: 'luo', 'swa' (or active_src_lang_short, active_tgt_lang_short)
        # Synthetic data from generator has 'original_src' and 'synthetic_tgt'
        synthetic_train_ds = synthetic_train_ds.rename_column("original_src", active_src_lang_short)
        synthetic_train_ds = synthetic_train_ds.rename_column("synthetic_tgt", active_tgt_lang_short)
        final_train_ds = concatenate_datasets([authentic_parallel_train_ds, synthetic_train_ds]).shuffle(seed=42)
    else:
        final_train_ds = authentic_parallel_train_ds.shuffle(seed=42)

    final_eval_ds = authentic_parallel_dev_ds
    if not final_eval_ds or len(final_eval_ds) == 0:
        # Create a minimal eval set from training if none exists
        if len(final_train_ds) > 20:  # Need at least some data to split
            logger.warning(
                "No evaluation dataset provided or created. Splitting 10 examples from training data for evaluation.")
            split = final_train_ds.train_test_split(test_size=min(10, len(final_train_ds) - 1),
                                                    shuffle=False)  # Minimal eval
            final_train_ds = split['train']
            final_eval_ds = split['test']
        else:
            logger.error("Not enough data for train/eval split. Evaluation will be problematic.")
            # Fallback: use a tiny part of train for eval if trainer requires it.
            # This is not ideal for actual model assessment.
            final_eval_ds = final_train_ds.select(range(min(1, len(final_train_ds))))

    logger.info(f"Total training samples: {len(final_train_ds)}")
    logger.info(f"Total evaluation samples: {len(final_eval_ds)}")
    if len(final_train_ds) == 0:
        logger.error("No training data available. Exiting.")
        return

    # --- 4. Character Vocabulary Building ---
    logger.info("Loading student tokenizer for character vocabulary building...")
    # Use the student's tokenizer to get subwords whose characters will form the char vocab
    # NLLB tokenizer needs src_lang for its internal state for proper tokenization if it's language-specific
    # For mT5, it's usually language-agnostic by default unless a task prefix is used.
    student_subword_tokenizer_for_char_vocab = AutoTokenizer.from_pretrained(
        STUDENT_CKPT,
        src_lang=active_src_lang_nllb,  # For NLLB student base
        # For mT5, src_lang might not be needed here, but doesn't hurt for NLLB compatibility
    )

    char_vocab_builder = CharacterVocabulary(special_tokens=SPECIAL_CHAR_TOKENS)
    # Collect all unique sentences from train and eval to build comprehensive char vocab
    all_text_for_char_vocab = []
    all_text_for_char_vocab.extend(final_train_ds[active_src_lang_short])
    all_text_for_char_vocab.extend(final_train_ds[active_tgt_lang_short])
    if final_eval_ds and len(final_eval_ds) > 0:
        all_text_for_char_vocab.extend(final_eval_ds[active_src_lang_short])
        all_text_for_char_vocab.extend(final_eval_ds[active_tgt_lang_short])

    char_vocab_builder.build_from_data(all_text_for_char_vocab, student_subword_tokenizer_for_char_vocab)

    # --- 5. Model Initialization ---
    logger.info("Loading teacher model for distillation...")
    # Teacher model is needed by DistillationTrainer
    # Load with BNB if large and on GPU
    use_bnb_for_teacher_in_trainer = (DEVICE == 'cuda' and "3.3B" in TEACHER_CKPT or "1.3B" in TEACHER_CKPT)
    if use_bnb_for_teacher_in_trainer:
        bnb_config_teacher_trainer = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        teacher_model_for_distillation = AutoModelForSeq2SeqLM.from_pretrained(
            TEACHER_CKPT,
            quantization_config=bnb_config_teacher_trainer,
            device_map="auto"
        )
        logger.info(f"Teacher model ({TEACHER_CKPT}) loaded with 4-bit BNB for DistillationTrainer.")
    else:
        teacher_model_for_distillation = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_CKPT)
        # No .to(DEVICE) here, trainer will handle it.
        logger.info(f"Teacher model ({TEACHER_CKPT}) loaded without BNB for DistillationTrainer.")

    logger.info(f"Initializing Character-Aware Student Model with base: {STUDENT_CKPT}")
    # Student tokenizer - this is the main tokenizer for the rest of the pipeline
    student_tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_CKPT,
        src_lang=active_src_lang_nllb,  # For NLLB student base
        tgt_lang=active_tgt_lang_nllb  # For NLLB student base
    )

    student_model = CharacterAwareMTModel(
        base_model_name_or_path=STUDENT_CKPT,
        char_vocab_size=len(char_vocab_builder),
        char_pad_idx=char_vocab_builder.get_pad_id(),
        char_emb_dim=CHAR_EMB_DIM,
        char_cnn_output_dim_ratio=0.25,  # e.g. 25% of d_model for char features
        freeze_base_model=False,  # Important: Fine-tune the student, including base
        freeze_base_embeddings=False,  # Allow subword embeddings of student to be trained
        use_bnb_quantization=DEVICE
    )
    student_model.to(DEVICE)  # Ensure student model is on the correct device if not using device_map in its __init__

    # --- 6. Prepare Datasets for Training ---
    logger.info("Creating NMTCharSubwordDataset for training...")
    train_torch_dataset = NMTCharSubwordDataset(
        hf_dataset=final_train_ds,
        subword_tokenizer=student_tokenizer,
        char_vocab=char_vocab_builder,
        max_subword_seq_len=MAX_SUBWORD_SEQ_LEN,
        max_char_len_per_subword_incl_special=MAX_CHAR_LEN_PER_SUBWORD,
        src_lang_key=active_src_lang_short,
        tgt_lang_key=active_tgt_lang_short
    )
    eval_torch_dataset = None
    if final_eval_ds and len(final_eval_ds) > 0:
        logger.info("Creating NMTCharSubwordDataset for evaluation...")
        eval_torch_dataset = NMTCharSubwordDataset(
            hf_dataset=final_eval_ds,
            subword_tokenizer=student_tokenizer,
            char_vocab=char_vocab_builder,
            max_subword_seq_len=MAX_SUBWORD_SEQ_LEN,
            max_char_len_per_subword_incl_special=MAX_CHAR_LEN_PER_SUBWORD,
            src_lang_key=active_src_lang_short,
            tgt_lang_key=active_tgt_lang_short
        )
    else:
        logger.warning("No evaluation dataset. Trainer will not perform evaluations during training.")

    # --- 7. Training Arguments and Collator ---
    # Make sure student_tokenizer has pad_token set if not already.
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token  # Common practice for many models
        student_model.config.pad_token_id = student_tokenizer.eos_token_id
        logger.info(f"Set student tokenizer pad_token to eos_token: {student_tokenizer.pad_token}")

    custom_data_collator = CustomDataCollator(
        subword_pad_token_id=student_tokenizer.pad_token_id,
        char_pad_token_id=char_vocab_builder.get_pad_id(),
        max_char_len_per_subword=MAX_CHAR_LEN_PER_SUBWORD
    )

    # For labels=-100 in DataCollatorForSeq2Seq, but our custom one handles it.
    # If using DataCollatorForSeq2Seq, it needs model for decoder_input_ids shifting.
    # data_collator = DataCollatorForSeq2Seq(tokenizer=student_tokenizer, model=student_model, label_pad_token_id=-100)

    output_path = Path("./outputs") / f"{active_src_lang_short}_{active_tgt_lang_short}_student"
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
        evaluation_strategy="steps" if eval_torch_dataset else "no",
        eval_steps=EVAL_STEPS if eval_torch_dataset else None,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,  # Essential for BLEU, etc.
        fp16=torch.cuda.is_available(),  # Enable FP16 if on GPU
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        load_best_model_at_end=True if eval_torch_dataset else False,
        metric_for_best_model="bleu" if eval_torch_dataset else None,
        report_to="tensorboard",  # or "wandb"
        dataloader_num_workers=2 if DEVICE == 'cuda' else 0,  # Can speed up data loading
        generation_max_length=MAX_SUBWORD_SEQ_LEN,  # For generation during evaluation
        # For NLLB, need to provide target lang ID for generation
        # This is usually handled by passing `forced_bos_token_id` to `model.generate`
        # The `compute_metrics_fn` will use the tokenizer, which if NLLB, needs this.
        # The trainer's predict_with_generate will call model.generate().
        # We can pass generation_config to Seq2SeqTrainingArguments.
        # generation_config = GenerationConfig(forced_bos_token_id=student_tokenizer.lang_code_to_id[active_tgt_lang_nllb])
        # This requires GenerationConfig to be imported.
        # Alternatively, the model's forward or generate method might need to be aware or use a generation_config.
        # For NLLB, the tokenizer usually prepares decoder_input_ids with the target lang code if used `as_target_tokenizer`.
        # Let's assume for now the tokenizer handles it for labels, and generation needs it explicitly.
        # The `predict_with_generate` in Trainer calls model.generate.
        # We might need to wrap the model or customize generate for NLLB target lang id.
        # For now, we rely on the tokenizer for metric computation to handle it.
        # If student is NLLB, its generate method can take `forced_bos_token_id`.
        # The training_args can accept a `generation_config` object.
    )
    if "nllb" in STUDENT_CKPT.lower():
        training_args.generation_config = student_model.config.generation_config
        training_args.generation_config.forced_bos_token_id = student_tokenizer.lang_code_to_id[active_tgt_lang_nllb]

    # --- 8. Initialize DistillationTrainer ---
    trainer = DistillationTrainer(
        teacher_model=teacher_model_for_distillation,
        model=student_model,
        args=training_args,
        train_dataset=train_torch_dataset,
        eval_dataset=eval_torch_dataset,
        tokenizer=student_tokenizer,  # Used for decoding predictions for metrics
        data_collator=custom_data_collator,
        compute_metrics=lambda p: compute_metrics_fn(p, student_tokenizer,final_eval_ds[active_src_lang_short] if final_eval_ds else None) \
            if eval_torch_dataset else None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] if eval_torch_dataset else [],
        kd_alpha=KD_ALPHA,
        rep_alpha=REP_ALPHA,
        # rep_layers_indices=PATIENT_REP_LAYERS_INDICES,
        # temperature=KD_TEMPERATURE
    )

    # --- 9. Train ---
    logger.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

    # --- 10. Evaluate and Save ---
    if eval_torch_dataset:
        logger.info("Evaluating final model on the evaluation set...")
        eval_results = trainer.evaluate(metric_key_prefix="final_eval")
        logger.info(f"Final evaluation results: {eval_results}")
        for key, value in eval_results.items():
            print(f"{key}: {value}")

    final_model_path = output_path / "final_character_aware_student_model"
    trainer.save_model(str(final_model_path))  # Saves the student_model (CharacterAwareMTModel)
    student_tokenizer.save_pretrained(str(final_model_path))  # Save tokenizer too
    logger.info(f"Final student model saved to {final_model_path}")

    # --- 11. Optional: Quantization of the base model part for deployment ---
    # Note: Quantizing the CharacterAwareMTModel directly with from_pretrained might be tricky.
    # This part assumes you might want to deploy only the base_model part after fine-tuning its weights.
    # If CharacterAwareMTModel's `save_pretrained` is fully compatible, this might work.
    # The original script did this on the output of trainer.save_model.
    try:
        logger.info(f"Attempting to load and quantize the saved base model from {final_model_path} (if applicable)")
        # If CharacterAwareMTModel saves its base_model in a 'base_model' subfolder:
        base_model_saved_path = final_model_path / "base_model"
        if base_model_saved_path.exists():

            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
                # For 8-bit, other params like bnb_8bit_quant_type can be set if needed.
            )
            # Load the base model specifically for quantization
            quantized_base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_saved_path,  # Path to the saved base_model component
                quantization_config=quant_config,
                device_map="auto"  # Or specific device
            )
            quantized_model_save_path = final_model_path / "base_model_int8"
            quantized_base_model.save_pretrained(str(quantized_model_save_path))
            student_tokenizer.save_pretrained(str(quantized_model_save_path))  # Save tokenizer with it
            logger.info(f"8-bit quantized base model saved to {quantized_model_save_path}")
        else:
            logger.warning(f"Base model not found at {base_model_saved_path} for quantization. "
                           "Quantization step skipped. This is expected if custom model doesn't save base separately in that exact path.")

    except Exception as e:
        logger.error(f"Error during quantization: {e}", exc_info=True)

    # --- 12. Example Inference (using the full CharacterAwareMTModel) ---
    try:
        logger.info("Loading saved CharacterAwareMTModel for inference test...")
        # Load the custom model using its from_pretrained method
        loaded_student_model = CharacterAwareMTModel.from_pretrained(str(final_model_path))
        loaded_student_model.to(DEVICE)
        loaded_student_model.eval()

        test_sentence_src = ""
        if final_eval_ds and len(final_eval_ds) > 0:
            test_sentence_src = final_eval_ds[0][active_src_lang_short]
        elif final_train_ds and len(final_train_ds) > 0:
            test_sentence_src = final_train_ds[0][active_src_lang_short]
        else:
            test_sentence_src = "Jothurwa welo." if active_src_lang_short == "luo" else "Example source sentence."

        logger.info(f"Test inference with sentence: '{test_sentence_src}'")

        # Prepare input for CharacterAwareMTModel
        inputs = student_tokenizer(test_sentence_src, return_tensors="pt", truncation=True,
                                   max_length=MAX_SUBWORD_SEQ_LEN)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        src_subword_strings_inf = student_tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist(),
                                                                          skip_special_tokens=False)
        source_char_ids_inf_list = []
        for sub_str in src_subword_strings_inf:
            source_char_ids_inf_list.append(
                torch.tensor(char_vocab_builder.encode_subword_string(sub_str, MAX_CHAR_LEN_PER_SUBWORD),
                             dtype=torch.long))

        # Collate this single example's char_ids (pad subword sequence dim)
        padded_char_ids_inf = torch.full(
            (1, input_ids.size(1), MAX_CHAR_LEN_PER_SUBWORD),
            fill_value=char_vocab_builder.get_pad_id(), dtype=torch.long
        )
        for i, char_tensor in enumerate(source_char_ids_inf_list):
            padded_char_ids_inf[0, i, :] = char_tensor
        source_char_ids_inf_tensor = padded_char_ids_inf.to(DEVICE)

        with torch.no_grad():
            generation_kwargs = {}
            if "nllb" in STUDENT_CKPT.lower():
                generation_kwargs["forced_bos_token_id"] = student_tokenizer.lang_code_to_id[active_tgt_lang_nllb]

            output_ids = loaded_student_model.base_model.generate(  # Call generate on the underlying base_model
                inputs_embeds=loaded_student_model(input_ids=input_ids, source_char_ids=source_char_ids_inf_tensor,
                                                   attention_mask=attention_mask).inputs_embeds,
                # Get the combined embeds
                attention_mask=attention_mask,  # Pass original attention mask
                max_length=MAX_SUBWORD_SEQ_LEN,
                num_beams=4,
                early_stopping=True,
                **generation_kwargs
            )
        translation = student_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        logger.info(f"Source ({active_src_lang_short}): {test_sentence_src}")
        logger.info(f"Translated ({active_tgt_lang_short}): {translation[0]}")

    except Exception as e:
        logger.error(f"Error during example inference: {e}", exc_info=True)

    logger.info(f"{PROJECT_NAME} pipeline complete.")
    logger.info(f"Final models and outputs are in: {output_path}")
    logger.info("Remember to check TensorBoard logs if report_to='tensorboard' was used.")


if __name__ == '__main__':
    # Before running main, ensure opustools-pkg is installed if USE_EXAMPLE_DATA is True
    # and necessary data files/directories are set up by the user if USE_EXAMPLE_DATA is False.
    # if USE_EXAMPLE_DATA or CREATE_DUMMY_DATA_IF_NONE_EXIST:  # Check if opus_read is needed
    #     try:
    #         subprocess.run(["opus_read", "--help"], capture_output=True, check=True)
    #         logger.info("opustools-pkg seems to be installed.")
    #     except (subprocess.CalledProcessError, FileNotFoundError):
    #         logger.error(
    #             "opustools-pkg is not installed or not found in PATH, but it's needed for example data or dummy data download might fail.")
    #         logger.error("Please install it: pip install opustools-pkg")
    #         # Decide if to exit or proceed if only dummy user data is used and opus_read is not strictly needed.
    #         # For now, we'll let it try and fail in download_opus_data if needed.

    main()