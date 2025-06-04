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

from peft import PeftModel
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
    EarlyStoppingCallback, GenerationConfig
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

BASE_PROJECT_DIR = Path("/content")
DATA_ROOT_DIR = BASE_PROJECT_DIR / "data"
USER_MONO_SRC_DIR = DATA_ROOT_DIR / "monolingual" / "collected"
USER_MONO_TGT_DIR = DATA_ROOT_DIR / "monolingual" / "processed"

PARALLEL_CSV_FILE = DATA_ROOT_DIR / "parallel" / "dholuo_swahili.csv"


USER_PARALLEL_TRAIN_SRC_FILE = DATA_ROOT_DIR / "parallel" / f"train_parallel.{SRC_LANG_SHORT}"
USER_PARALLEL_TRAIN_TGT_FILE = DATA_ROOT_DIR / "parallel" / f"train_parallel.{TGT_LANG_SHORT}"
# If you create dev files from your CSV:
USER_PARALLEL_DEV_SRC_FILE = DATA_ROOT_DIR / "parallel" / f"dev_parallel.{SRC_LANG_SHORT}"
USER_PARALLEL_DEV_TGT_FILE = DATA_ROOT_DIR / "parallel" / f"dev_parallel.{TGT_LANG_SHORT}"


USER_MONO_SRC_DIR.mkdir(parents=True, exist_ok=True)
USER_MONO_TGT_DIR.mkdir(parents=True, exist_ok=True)
(DATA_ROOT_DIR / "parallel").mkdir(parents=True, exist_ok=True)

# Model checkpoints
TEACHER_CKPT = 'facebook/nllb-200-distilled-600M'
STUDENT_CKPT = 'google/mt5-small'
STUDENT_USE_BNB_FOR_PEFT = True # QLoRA-style training of student


# Task Prefixes for Bidirectional Student Model
TASK_PREFIX_LUO_TO_SWA = "translate Dholuo to Swahili: "
TASK_PREFIX_SWA_TO_LUO = "translate Swahili to Dholuo: "

# Character CNN parameters
CHAR_EMB_DIM = 64  # Reduced for memory
CHAR_CNN_KERNEL_SIZES = [2, 3, 4]  # Fewer/smaller kernels
CHAR_CNN_NUM_FILTERS = 64  # Reduced number of filters
CHAR_CNN_OUTPUT_DIM_RATIO = 0.25 # CharCNN output dim = d_model * this ratio


# Special character tokens
CHAR_PAD_TOKEN = "<CPAD>"  # Character PAD
CHAR_UNK_TOKEN = "<CUNK>"  # Character UNK
CHAR_SOW_TOKEN = "<SOW>"  # Start of Word/Subword
CHAR_EOW_TOKEN = "<EOW>"  # End of Word/Subword
SPECIAL_CHAR_TOKENS = [CHAR_PAD_TOKEN, CHAR_UNK_TOKEN, CHAR_SOW_TOKEN, CHAR_EOW_TOKEN]

# Training hyperparameters
MAX_SUBWORD_SEQ_LEN = 128
MAX_CHAR_LEN_PER_SUBWORD = 16  # Max characters per subword token (for padding char sequences), INCLUDES SOW/EOW
BATCH_SIZE = 16  # For Colab Free GPU. For T4, can try 16.
GRAD_ACCUM_STEPS = 4  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS (32)
LEARNING_RATE = 3e-4  # Slightly higher for smaller models / from-scratch components
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5  # Increase for real training (e.g., 10-20)
KD_ALPHA = 0.7
REP_ALPHA = 0.3
KD_TEMPERATURE = 2.0
PATIENT_REP_LAYERS_INDICES = [-1] # Distill from last encoder hidden layer
PEFT_LORA_R = 8
PEFT_LORA_ALPHA = 16
PEFT_LORA_DROPOUT = 0.05

# Evaluation & Logging
EVAL_STEPS = 250  # Evaluate more frequently for low-resource
LOGGING_STEPS = 50
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
EARLY_STOPPING_PATIENCE = 3

# Metrics - initialize once
bleu_metric = evaluate.load('sacrebleu')
chrf_metric = evaluate.load('chrf')  # Returns score 0-1 by default
bertscore_metric = evaluate.load('bertscore')
comet_metric = None  # Initialize later due to potential download/resource issues
BLEURT_CKPT = 'BLEURT-20'  # General purpose checkpoint
bleurt_metric = None

STUDENT_USE_BNB = True

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

    logger.info("file processing for multitask training.")
    active_src_lang_short_main, active_tgt_lang_short_main = SRC_LANG_SHORT, TGT_LANG_SHORT  # luo, swa
    active_src_nllb_main, active_tgt_nllb_main = SRC_LANG_NLLB, TGT_LANG_NLLB  # luo_Latn, swa_Latn

    mono_lang1_dir = USER_MONO_SRC_DIR  # Dholuo mono
    mono_lang2_dir = USER_MONO_TGT_DIR  # Swahili mono

    para_lang1_file = USER_PARALLEL_TRAIN_SRC_FILE  # train.luo
    para_lang2_file = USER_PARALLEL_TRAIN_TGT_FILE  # train.swa
    para_dev_lang1_file = USER_PARALLEL_DEV_SRC_FILE  # dev.luo
    para_dev_lang2_file = USER_PARALLEL_DEV_TGT_FILE  # dev.swa

    # --- 1. Text Preprocessor (Generic for both languages) ---
    text_preprocessor = TextPreprocessor(
        src_lang_short_code=active_src_lang_short_main,  # For default behavior if lang not specific
        tgt_lang_short_code=active_tgt_lang_short_main
    )
    file_processor = FileProcessor(text_preprocessor)

    # --- 2. Load Raw Data ---
    # Monolingual data for language 1 (e.g., Dholuo or English)
    mono_l1_sentences = file_processor.load_and_process_monolingual_dir(
        mono_lang1_dir, active_src_lang_short_main
    )
    # Monolingual data for language 2 (e.g., Swahili or German)
    mono_l2_sentences = file_processor.load_and_process_monolingual_dir(
        mono_lang2_dir, active_tgt_lang_short_main
    )

    # Parallel data (content is lang1 <-> lang2)
    # FileProcessor returns dict with actual lang keys e.g. {'luo': ..., 'swa': ...} or {'en':..., 'de':...}
    parallel_train_raw_pairs = file_processor.load_and_process_parallel_files(
        para_lang1_file, para_lang2_file, active_src_lang_short_main, active_tgt_lang_short_main
    )
    parallel_dev_raw_pairs = []
    if para_dev_lang1_file and para_dev_lang2_file and para_dev_lang1_file.exists() and para_dev_lang2_file.exists():
        parallel_dev_raw_pairs = file_processor.load_and_process_parallel_files(
            para_dev_lang1_file, para_dev_lang2_file, active_src_lang_short_main, active_tgt_lang_short_main
        )

    # --- 3. Create Multitask Dataset from Parallel Data ---
    multitask_parallel_train_examples = []
    for pair in parallel_train_raw_pairs:
        l1_text = pair[active_src_lang_short_main]  # e.g., Dholuo text
        l2_text = pair[active_tgt_lang_short_main]  # e.g., Swahili text

        # Direction 1: lang1 -> lang2 (e.g., Luo -> Swahili)
        multitask_parallel_train_examples.append({
            "student_input_text": TASK_PREFIX_LUO_TO_SWA + l1_text,
            "target_text": l2_text,
            "teacher_input_text": l1_text,  # Unprefixed for teacher
            "teacher_src_nllb": active_src_nllb_main,  # NLLB code for l1
            "teacher_tgt_nllb": active_tgt_nllb_main,  # NLLB code for l2
            "unprefixed_src_for_comet": l1_text  # For metrics
        })
        # Direction 2: lang2 -> lang1 (e.g., Swahili -> Luo)
        multitask_parallel_train_examples.append({
            "student_input_text": TASK_PREFIX_LUO_TO_SWA + l2_text,
            "target_text": l1_text,
            "teacher_input_text": l2_text,  # Unprefixed for teacher
            "teacher_src_nllb": active_tgt_nllb_main,  # NLLB code for l2
            "teacher_tgt_nllb": active_src_nllb_main,  # NLLB code for l1
            "unprefixed_src_for_comet": l2_text  # For metrics
        })

    multitask_parallel_dev_examples = []
    for pair in parallel_dev_raw_pairs:
        l1_text = pair[active_src_lang_short_main]
        l2_text = pair[active_tgt_lang_short_main]
        multitask_parallel_dev_examples.append({
            "student_input_text": TASK_PREFIX_LUO_TO_SWA + l1_text,
            "target_text": l2_text, "teacher_input_text": l1_text,
            "teacher_src_nllb": active_src_nllb_main, "teacher_tgt_nllb": active_tgt_nllb_main,
            "unprefixed_src_for_comet": l1_text
        })
        multitask_parallel_dev_examples.append({
            "student_input_text": TASK_PREFIX_SWA_TO_LUO + l2_text,
            "target_text": l1_text, "teacher_input_text": l2_text,
            "teacher_src_nllb": active_tgt_nllb_main, "teacher_tgt_nllb": active_src_nllb_main,
            "unprefixed_src_for_comet": l2_text
        })

    authentic_parallel_train_ds = HFDataset.from_list(multitask_parallel_train_examples)
    authentic_parallel_dev_ds = HFDataset.from_list(
        multitask_parallel_dev_examples) if multitask_parallel_dev_examples else None

    # --- 4. Synthetic Data Generation for Both Directions ---
    all_synthetic_examples = []
    global sbert_model  # Ensure sbert_model is accessible

    # Direction 1: L1_mono -> L2_synthetic (e.g., Luo -> Swahili)
    if mono_l1_sentences:
        logger.info(f"Generating synthetic data for {active_src_lang_short_main} -> {active_tgt_lang_short_main}")
        synth_gen_l1_to_l2 = SyntheticDataGeneratorAndFilter(
            TEACHER_CKPT, TEACHER_CKPT,
            sbert_model_instance=sbert_model,  # Pass loaded sbert
            current_src_lang_nllb_code=active_src_nllb_main,
            current_tgt_lang_nllb_code=active_tgt_nllb_main,
            device=DEVICE, use_bnb_for_teacher_pipeline=True
        )
        raw_synth_pairs_l1_to_l2 = synth_gen_l1_to_l2.generate_synthetic_data(mono_l1_sentences,
                                                                              batch_size=BATCH_SIZE * 2)
        if raw_synth_pairs_l1_to_l2:
            filtered_synth_l1_to_l2 = synth_gen_l1_to_l2.apply_all_filters(
                HFDataset.from_list(raw_synth_pairs_l1_to_l2))
            for item in filtered_synth_l1_to_l2:  # item is {'original_src': ..., 'synthetic_tgt': ...}
                all_synthetic_examples.append({
                    "student_input_text": TASK_PREFIX_LUO_TO_SWA + item["original_src"],
                    "target_text": item["synthetic_tgt"],
                    "teacher_input_text": item["original_src"],
                    "teacher_src_nllb": active_src_nllb_main,
                    "teacher_tgt_nllb": active_tgt_nllb_main,
                    "unprefixed_src_for_comet": item["original_src"]
                })
        del synth_gen_l1_to_l2.teacher_model;
        del synth_gen_l1_to_l2;
        gc.collect();
        torch.cuda.empty_cache()

    # Direction 2: L2_mono -> L1_synthetic (e.g., Swahili -> Luo)
    if mono_l2_sentences:
        logger.info(f"Generating synthetic data for {active_tgt_lang_short_main} -> {active_src_lang_short_main}")
        synth_gen_l2_to_l1 = SyntheticDataGeneratorAndFilter(
            TEACHER_CKPT, TEACHER_CKPT,
            sbert_model_instance=sbert_model,
            current_src_lang_nllb_code=active_tgt_nllb_main,  # Swapped
            current_tgt_lang_nllb_code=active_src_nllb_main,  # Swapped
            device=DEVICE, use_bnb_for_teacher_pipeline=True
        )
        raw_synth_pairs_l2_to_l1 = synth_gen_l2_to_l1.generate_synthetic_data(mono_l2_sentences,
                                                                              batch_size=BATCH_SIZE * 2)
        if raw_synth_pairs_l2_to_l1:
            filtered_synth_l2_to_l1 = synth_gen_l2_to_l1.apply_all_filters(
                HFDataset.from_list(raw_synth_pairs_l2_to_l1))
            for item in filtered_synth_l2_to_l1:
                all_synthetic_examples.append({
                    "student_input_text": TASK_PREFIX_SWA_TO_LUO + item["original_src"],
                    "target_text": item["synthetic_tgt"],
                    "teacher_input_text": item["original_src"],
                    "teacher_src_nllb": active_tgt_nllb_main,  # Swapped
                    "teacher_tgt_nllb": active_src_nllb_main,  # Swapped
                    "unprefixed_src_for_comet": item["original_src"]
                })
        del synth_gen_l2_to_l1.teacher_model
        del synth_gen_l2_to_l1
        gc.collect()
        torch.cuda.empty_cache()

    # --- 5. Combine all data and create final train/eval splits ---
    final_train_ds_list = [authentic_parallel_train_ds]
    if all_synthetic_examples:
        final_train_ds_list.append(HFDataset.from_list(all_synthetic_examples))

    if not final_train_ds_list[0] and (len(final_train_ds_list) == 1 or not final_train_ds_list[1]):
        logger.error("No authentic parallel data and no synthetic data. Cannot train.")
        return

    final_train_ds = concatenate_datasets(final_train_ds_list).shuffle(seed=42)

    final_eval_ds = authentic_parallel_dev_ds  # Use dev set from bidirectional parallel data
    if not final_eval_ds or len(final_eval_ds) == 0:
        if len(final_train_ds) > 40:  # Need more for multitask eval
            logger.warning("No evaluation dataset. Splitting from combined training data for multitask evaluation.")
            split_size = min(max(20, int(len(final_train_ds) * 0.01)), 1000)
            if len(final_train_ds) - split_size < 1: split_size = len(final_train_ds) - 1
            if split_size > 0:
                split = final_train_ds.train_test_split(test_size=split_size, shuffle=True,
                                                        seed=42)  # Shuffle before split
                final_train_ds, final_eval_ds = split['train'], split['test']
        else:
            logger.error("Not enough data for train/eval split. Evaluation will use training data.")
            final_eval_ds = final_train_ds.select(range(min(max(1, len(final_train_ds) // 2), 20)))  # Tiny eval

    logger.info(f"Multitask Training: Total training samples: {len(final_train_ds)}")
    logger.info(f"Multitask Training: Total evaluation samples: {len(final_eval_ds)}")
    if len(final_train_ds) == 0: logger.error("No training data. Exiting."); return

    # --- 6. Joint Character Vocabulary ---
    logger.info("Building joint character vocabulary for multitask model...")
    # For student (mT5), tokenizer doesn't typically need src_lang for general tokenization
    # unless it's specifically an NLLB-family tokenizer being used as student.
    # For vocab building, any consistent tokenizer that gives subword strings is okay.
    # NLLB tokenizer with a default lang (like 'eng_Latn') is fine for getting chars of its vocab.
    tokenizer_for_char_vocab_build = AutoTokenizer.from_pretrained(STUDENT_CKPT,
                                                                   src_lang="eng_Latn" if "nllb" in STUDENT_CKPT.lower() else None)

    texts_for_char_vocab = mono_l1_sentences + mono_l2_sentences
    for ex in final_train_ds: texts_for_char_vocab.append(ex['teacher_input_text']); texts_for_char_vocab.append(
        ex['target_text'])
    if final_eval_ds:
        for ex in final_eval_ds: texts_for_char_vocab.append(ex['teacher_input_text']); texts_for_char_vocab.append(
            ex['target_text'])

    char_vocab_builder = CharacterVocabulary(special_tokens=SPECIAL_CHAR_TOKENS)
    char_vocab_builder.build_from_data(
        texts_for_char_vocab,
        tokenizer_for_char_vocab_build,
        task_prefixes=[TASK_PREFIX_LUO_TO_SWA, TASK_PREFIX_SWA_TO_LUO]
    )
    del texts_for_char_vocab, tokenizer_for_char_vocab_build;
    gc.collect()

    # --- 7. Model Initialization (Single Student, Teacher for Distillation) ---
    logger.info(f"Loading teacher model ({TEACHER_CKPT}) for distillation.")
    use_bnb_for_teacher_trainer = (DEVICE == 'cuda' and any(s in TEACHER_CKPT for s in ["3.3B", "1.3B", "600M"]))
    teacher_bnb_config = None
    if use_bnb_for_teacher_trainer:
        teacher_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True)

    teacher_model_for_distillation = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER_CKPT,
        quantization_config=teacher_bnb_config,
        device_map="auto" if teacher_bnb_config else None
    )
    # The teacher's tokenizer will be loaded dynamically inside DistillationTrainer based on batch's teacher_src_nllb_code
    if not teacher_bnb_config: teacher_model_for_distillation.to(DEVICE)
    logger.info(f"Teacher model for distillation loaded ({'BNB' if teacher_bnb_config else 'standard'}).")

    # Student tokenizer (e.g., mT5 tokenizer, generally lang-agnostic for tokenization itself)
    # NLLB student tokenizer would need src_lang/tgt_lang if STUDENT_CKPT is NLLB.
    # For mT5, the prefixes handle direction.
    student_tokenizer_main = AutoTokenizer.from_pretrained(STUDENT_CKPT)
    if student_tokenizer_main.pad_token is None:
        student_tokenizer_main.pad_token = student_tokenizer_main.eos_token
        logger.info(f"Set student tokenizer pad_token to eos_token: {student_tokenizer_main.pad_token}")

    student_model = CharacterAwareMTModel(
        base_model_name_or_path=STUDENT_CKPT,
        char_vocab_size=len(char_vocab_builder),
        char_pad_idx=char_vocab_builder.get_pad_id(),
        char_emb_dim=CHAR_EMB_DIM,
        char_cnn_output_dim_ratio=CHAR_CNN_OUTPUT_DIM_RATIO,
        use_bnb_quantization_for_base_before_peft=(STUDENT_USE_BNB_FOR_PEFT and DEVICE == 'cuda'),
        peft_lora_r=PEFT_LORA_R, peft_lora_alpha=PEFT_LORA_ALPHA, peft_lora_dropout=PEFT_LORA_DROPOUT
    )
    if not (STUDENT_USE_BNB_FOR_PEFT and DEVICE == 'cuda'):
        student_model.to(DEVICE)
    else:  # Ensure custom parts are on device if base used device_map
        if student_model.char_cnn: student_model.char_cnn.to(DEVICE)
        if student_model.projection_layer: student_model.projection_layer.to(DEVICE)

    # --- 8. Prepare Datasets for Training ---
    logger.info("Creating NMTCharSubwordDataset for multitask training...")
    # Teacher tokenizer is needed by dataset to prepare teacher_input_ids.
    # This is tricky if teacher is NLLB and tokenizer needs src_lang.
    # We simplified by passing text to trainer.
    # Here, student_tokenizer is passed, as it tokenizes the student_input_text (with prefix)
    train_torch_dataset = NMTCharSubwordDataset(
        hf_dataset=final_train_ds,
        student_subword_tokenizer=student_tokenizer_main,
        teacher_subword_tokenizer=None,  # Teacher tokenization handled in DistillationTrainer
        char_vocab=char_vocab_builder,
        max_subword_seq_len=MAX_SUBWORD_SEQ_LEN,
        max_char_len_per_subword_incl_special=MAX_CHAR_LEN_PER_SUBWORD,
        # Columns are now "student_input_text", "target_text", "teacher_input_text", etc.
        src_data_column_name="student_input_text",  # This has the prefix
        tgt_data_column_name="target_text"
    )
    eval_torch_dataset, eval_source_texts_for_comet = None, None
    if final_eval_ds and len(final_eval_ds) > 0:
        eval_torch_dataset = NMTCharSubwordDataset(
            hf_dataset=final_eval_ds, student_subword_tokenizer=student_tokenizer_main,
            teacher_subword_tokenizer=None, char_vocab=char_vocab_builder,
            max_subword_seq_len=MAX_SUBWORD_SEQ_LEN, max_char_len_per_subword_incl_special=MAX_CHAR_LEN_PER_SUBWORD,
            src_data_column_name="student_input_text", tgt_data_column_name="target_text"
        )
        eval_source_texts_for_comet = final_eval_ds["teacher_input_text"]  # Unprefixed source for COMET

    # --- 9. Training Arguments and Collator ---
    custom_data_collator = CustomDataCollator(
        student_tokenizer=student_tokenizer_main,  # Pass tokenizer for pad_token_id
        char_pad_token_id=char_vocab_builder.get_pad_id(),
        max_char_len_per_subword=MAX_CHAR_LEN_PER_SUBWORD
    )

    output_dir_multitask = Path("./outputs_multitask_student")
    output_dir_multitask.mkdir(parents=True, exist_ok=True)

    # Generation config for evaluation (student model)
    # For mT5, specific lang codes in forced_bos_token_id are not standard.
    # Generation for mT5 is usually controlled by the prefix in the input.
    # If STUDENT_CKPT were NLLB, then we would set forced_bos_token_id based on the target of the eval example.
    # This is complex for compute_metrics with mixed directions.
    # Simplification: rely on prefix for mT5. If NLLB student, this needs more careful handling in compute_metrics.
    generation_config_student = GenerationConfig(
        max_length=MAX_SUBWORD_SEQ_LEN, num_beams=4, early_stopping=True,
    )
    # If STUDENT_CKPT is NLLB, this would need to be set dynamically in compute_metrics based on the target lang of the batch.
    # Since student is mT5, we don't set forced_bos_token_id globally.

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir_multitask),
        evaluation_strategy="steps" if eval_torch_dataset else "no",
        eval_steps=EVAL_STEPS if eval_torch_dataset else None,
        logging_strategy="steps", logging_steps=LOGGING_STEPS,
        save_strategy="steps", save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        fp16=(DEVICE == 'cuda'),
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        load_best_model_at_end=True if eval_torch_dataset else False,
        metric_for_best_model="bleu" if eval_torch_dataset else None,  # bleu is generic here
        report_to="tensorboard",
        dataloader_num_workers=2 if DEVICE == 'cuda' else 0,  # Be careful on Colab with num_workers > 0
        generation_config=generation_config_student,
        # For gradient_checkpointing with PEFT and BNB, need to be careful with model wrapping
        # gradient_checkpointing=True, # If memory is an issue, but ensure compatibility
    )

    trainer = DistillationTrainer(
        teacher_model=teacher_model_for_distillation,
        model=student_model,  # This is the CharacterAwareMTModel instance
        args=training_args,
        train_dataset=train_torch_dataset,
        eval_dataset=eval_torch_dataset,
        tokenizer=student_tokenizer_main,  # For decoding preds in compute_metrics
        data_collator=custom_data_collator,
        compute_metrics=lambda p: compute_metrics_fn(p, student_tokenizer_main, eval_source_texts_for_comet) \
            if eval_torch_dataset else None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] if eval_torch_dataset else [],
        kd_alpha=KD_ALPHA, rep_alpha=REP_ALPHA,
        rep_layers_indices=PATIENT_REP_LAYERS_INDICES, temperature=KD_TEMPERATURE
    )

    # --- 10. Train ---
    logger.info("Starting multitask training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during multitask training: {e}", exc_info=True)
        raise

    # --- 11. Evaluate and Save ---
    if eval_torch_dataset:
        logger.info("Evaluating final multitask model...")
        eval_results = trainer.evaluate(metric_key_prefix="final_eval_multitask")
        logger.info(f"Final multitask evaluation results: {eval_results}")
        for key, value in eval_results.items(): print(f"Multitask - {key}: {value}")

    final_model_peft_path = output_dir_multitask / "final_multitask_student_peft_adapter"
    trainer.save_model(str(final_model_peft_path))  # Saves PEFT adapter & custom layers
    student_tokenizer_main.save_pretrained(str(final_model_peft_path))
    logger.info(f"Final multitask student PEFT model saved to {final_model_peft_path}")

    # --- 12. Merge PEFT and Quantize (Optional) ---
    try:
        logger.info(
            f"Loading base student model ({STUDENT_CKPT}) for merging PEFT adapter from {final_model_peft_path}")
        base_model_for_merge = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_CKPT).to(DEVICE)

        peft_model_for_merge = PeftModel.from_pretrained(base_model_for_merge, str(final_model_peft_path))
        peft_model_for_merge.eval()
        merged_model = peft_model_for_merge.merge_and_unload()
        logger.info("Successfully merged PEFT adapter into base model for multitask student.")

        merged_model_save_path = output_dir_multitask / "final_multitask_student_merged_model"
        merged_model.save_pretrained(str(merged_model_save_path))
        student_tokenizer_main.save_pretrained(str(merged_model_save_path))
        logger.info(f"Merged multitask student model saved to {merged_model_save_path}")

        if DEVICE == 'cuda':
            logger.info("Quantizing the merged multitask model to 8-bit...")
            quant_config_bnb = BitsAndBytesConfig(load_in_8bit=True)
            quantized_merged_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(merged_model_save_path),
                quantization_config=quant_config_bnb, device_map="auto"
            )
            quantized_save_path = output_dir_multitask / "final_multitask_student_merged_quantized_int8"
            quantized_merged_model.save_pretrained(str(quantized_save_path))
            student_tokenizer_main.save_pretrained(str(quantized_save_path))
            logger.info(f"8-bit quantized merged multitask model saved to {quantized_save_path}")
            del quantized_merged_model
        del base_model_for_merge, peft_model_for_merge, merged_model
    except Exception as e:
        logger.error(f"Error during PEFT merging/quantization for multitask model: {e}", exc_info=True)

    # --- 13. Example Inference for Multitask Model ---
    try:
        logger.info("Loading merged multitask student model for inference test...")
        # For inference, use the merged model (or the PEFT model directly before merging)
        # If CharacterAwareMTModel architecture is needed, then load CharCNN/Projection separately
        # and attach to the merged_model (if it's just the transformer part).
        # Here, we'll use the merged transformer part directly.

        # Load the non-quantized merged model for inference test
        inference_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(output_dir_multitask / "final_multitask_student_merged_model")).to(DEVICE)
        inference_tokenizer = AutoTokenizer.from_pretrained(
            str(output_dir_multitask / "final_multitask_student_merged_model"))
        inference_model.eval()

        test_luo_sentence = "An gima ber miwuoro."  # Example Dholuo
        test_swa_sentence = "Hii ni habari njema sana."  # Example Swahili

        # Test Luo -> Swahili
        input_text_l2s = TASK_PREFIX_LUO_TO_SWA + test_luo_sentence
        inputs_l2s = inference_tokenizer(input_text_l2s, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs_l2s = inference_model.generate(**inputs_l2s, max_length=MAX_SUBWORD_SEQ_LEN, num_beams=4)
        translation_l2s = inference_tokenizer.decode(outputs_l2s[0], skip_special_tokens=True)
        logger.info(
            f"Luo->Swahili | Source: '{test_luo_sentence}' | Prefixed Input: '{input_text_l2s}' | Translated: '{translation_l2s}'")

        # Test Swahili -> Luo
        input_text_s2l = TASK_PREFIX_SWA_TO_LUO + test_swa_sentence
        inputs_s2l = inference_tokenizer(input_text_s2l, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs_s2l = inference_model.generate(**inputs_s2l, max_length=MAX_SUBWORD_SEQ_LEN, num_beams=4)
        translation_s2l = inference_tokenizer.decode(outputs_s2l[0], skip_special_tokens=True)
        logger.info(
            f"Swa->Luo | Source: '{test_swa_sentence}' | Prefixed Input: '{input_text_s2l}' | Translated: '{translation_s2l}'")

        del inference_model

    except Exception as e:
        logger.error(f"Error during multitask inference test: {e}", exc_info=True)

    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    logger.info(f"{PROJECT_NAME} (multitask single student) pipeline complete.")


if __name__ == '__main__':
    main()
