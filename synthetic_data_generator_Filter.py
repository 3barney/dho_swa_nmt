import logging, torch, evaluate, torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Dict, Optional, Union

from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig,
)
from transformers.pipelines.pt_utils import KeyDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MAX_SUBWORD_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _to_dict(out):
    # pipeline returns [ {…} ] – unwrap it
    return out[0] if isinstance(out, list) else out


class SyntheticDataGeneratorAndFilter:
    def __init__(
        self,
        teacher_model_name_or_path: str,
        teacher_tokenizer_name_or_path: str,
        src_lang_nllb_code: str,
        tgt_lang_nllb_code: str,
        sbert_model_instance: Optional[SentenceTransformer] = None,
        device: str = DEVICE,
        fp16: bool = True,
    ):
        self.device = device
        self.src_lang = src_lang_nllb_code
        self.tgt_lang = tgt_lang_nllb_code

        logger.info("Loading NLLB teacher model …")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.teacher = AutoModelForSeq2SeqLM.from_pretrained(
            teacher_model_name_or_path,
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16 if fp16 else None,
            device_map="auto",
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_tokenizer_name_or_path, src_lang=self.src_lang, use_fast=True
        )

        self.fwd_pipe = pipeline(
            "translation",
            model=self.teacher,
            tokenizer=self.tokenizer,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            # device=0 if self.device == "cuda" else -1,
        )
        logger.info("Teacher pipeline initialised (GPU=%s).", self.device == "cuda")

        self.sbert = sbert_model_instance
        logger.info("SBERT model %sloaded.", "" if self.sbert else "NOT ")

        self.chrf = evaluate.load("chrf")

    def generate_synthetic_data(
        self, src_sentences: Union[List[str], Dataset], batch_size: int = 32, max_new_tokens: int = 400
    ) -> Dataset:
        """
        Translate Dholuo → Swahili using the GPU-backed pipeline
        and return an HF-Dataset with columns {original_src, synthetic_tgt}.
        """
        if not isinstance(src_sentences, Dataset):
            src_ds = Dataset.from_dict({"text": src_sentences})
        else:
            src_ds = src_sentences.rename_column(src_sentences.column_names[0], "text")

        logger.info("Generating %d synthetic sentences …", len(src_ds))

        out_tgts = []
        # KeyDataset streams one column lazily; tqdm tracks progress
        for out in tqdm(
                self.fwd_pipe(KeyDataset(src_ds, "text"),
                              batch_size=batch_size,
                              max_new_tokens=max_new_tokens,
                              truncation=True),
                total=len(src_ds),
                desc="Synthetic generation",
        ): out_tgts.append(_to_dict(out)["translation_text"])

        return src_ds.add_column("synthetic_tgt", out_tgts).rename_column("text", "original_src")

    def filter_round_trip(self, ds: Dataset, threshold: float = 0.45, batch_size: int = 32) -> Dataset:
        """
        Back-translate synthetic Swahili → Dholuo in batches, keep pairs with chrF ≥ threshold.
        """
        logger.info("RTT filter (chrF ≥ %.2f).  Input=%d pairs", threshold, len(ds))

        # A lightweight tokenizer reuse with src=tgt swap
        back_tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer.name_or_path, src_lang=self.tgt_lang, use_fast=True
        )
        back_pipe = pipeline(
            "translation",
            model=self.teacher,
            tokenizer=back_tokenizer,
            src_lang=self.tgt_lang,
            tgt_lang=self.src_lang,
            # device=0 if self.device == "cuda" else -1,
        )

        back_trans = []
        for out in tqdm(
                back_pipe(KeyDataset(ds, "synthetic_tgt"),
                          batch_size=batch_size,
                          truncation=True),
                total=len(ds),
                desc="Back-translate",
        ): back_trans.append(_to_dict(out)["translation_text"])

        chrf_scores = [
            self.chrf.compute(predictions=[bt], references=[[src]])["score"]
            for src, bt in zip(ds["original_src"], back_trans)
        ]

        keep_idx = [i for i, s in enumerate(chrf_scores) if s >= threshold]
        logger.info("RTT kept %d / %d", len(keep_idx), len(ds))
        return ds.select(keep_idx)

    def filter_semantic_similarity(self, ds: Dataset, threshold: float = 0.75, batch_size: int = 64) -> Dataset:
        if self.sbert is None:
            raise ValueError("SBERT model not provided.")
        logger.info("Semantic filter (cos ≥ %.2f).  Input=%d", threshold, len(ds))

        with torch.inference_mode():
            emb_src = self.sbert.encode(
                ds["original_src"], convert_to_tensor=True, device=self.device, batch_size=batch_size
            )
            emb_tgt = self.sbert.encode(
                ds["synthetic_tgt"], convert_to_tensor=True, device=self.device, batch_size=batch_size
            )
        sims = F.cosine_similarity(emb_src, emb_tgt).cpu().tolist()

        keep_idx = [i for i, sim in enumerate(sims) if sim >= threshold]
        logger.info("Semantic kept %d / %d", len(keep_idx), len(ds))
        return ds.select(keep_idx)

    def filter_lm_fluency(
        self, ds: Dataset, lang_key: str, threshold: float = -3.5, batch_size: int = 32
    ) -> Dataset:
        check_lang = self.tgt_lang if lang_key == "synthetic_tgt" else self.src_lang
        lm_tok = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path, src_lang=check_lang, use_fast=True)

        neg_losses: List[float] = []
        for i in tqdm(range(0, len(ds), batch_size), desc=f"LM-fluency({lang_key})"):
            batch = ds[lang_key][i : i + batch_size]
            inputs = lm_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SUBWORD_SEQ_LEN).to(
                self.device
            )
            labels = inputs.input_ids.clone()
            labels[labels == lm_tok.pad_token_id] = -100
            with torch.inference_mode():
                loss = self.teacher(**inputs, labels=labels).loss.item()
            neg_losses.extend([-loss] * len(batch))

        keep_idx = [i for i, nloss in enumerate(neg_losses) if nloss >= threshold]
        logger.info("LM-fluency kept %d / %d", len(keep_idx), len(ds))
        return ds.select(keep_idx)

    def apply_all_filters(
        self,
        synthetic_ds: Dataset,
        rtt_th: float = 0.45,
        sem_th: float = 0.75,
        lm_th: float = -3.5,
    ) -> Dataset:
        logger.info("Applying full filter cascade …")
        ds = self.filter_round_trip(synthetic_ds, threshold=rtt_th)
        if len(ds) == 0:
            return ds
        ds = self.filter_semantic_similarity(ds, threshold=sem_th)
        if len(ds) == 0:
            return ds
        ds = self.filter_lm_fluency(ds, lang_key="synthetic_tgt", threshold=lm_th)
        logger.info("Final dataset size: %d", len(ds))
        return ds
