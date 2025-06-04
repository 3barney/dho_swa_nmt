import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Seq2SeqTrainer
from typing import List, Tuple, Dict, Optional

from tqdm.auto import tqdm
from typing import List, Dict
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_SUBWORD_SEQ_LEN=128

# -----------------------
# DISTILLATION TRAINER (KD + Patient Representation Distillation)
# -----------------------
class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self,
                 teacher_model: nn.Module,
                 kd_alpha: float = 0.5,
                 rep_alpha: float = 0.5,
                 rep_layers_indices: Optional[List[int]] = None,  # e.g. [-1] for last encoder layer
                 temperature: float = 2.0,  # For softening teacher logits
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model.to(self.args.device)  # Ensure teacher is on the same device as student
        self.teacher.eval()  # Teacher should always be in eval mode

        self.kd_alpha = kd_alpha
        self.rep_alpha = rep_alpha
        self.rep_layers_indices = rep_layers_indices if rep_layers_indices is not None else [
            -1]  # Default to last layer
        self.temperature = temperature

        self.mse_loss = nn.MSELoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")  # 'batchmean' averages over batch
        logger.info(
            f"DistillationTrainer initialized with kd_alpha={kd_alpha}, rep_alpha={rep_alpha}, T={temperature}, rep_layers={self.rep_layers_indices}")

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard NLL loss from student
        # Note: The `model` here is the `CharacterAwareMTModel`
        # `inputs` will contain 'input_ids', 'attention_mask', 'labels', and 'source_char_ids'
        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "source_char_ids": inputs["source_char_ids"],
            "labels": inputs.get("labels")  # Might be None during eval if not passing labels
        }
        student_outputs = model(**student_inputs, output_hidden_states=True, output_attentions=False, return_dict=True)
        loss_nll = student_outputs.loss

        loss_kd = torch.tensor(0.0, device=loss_nll.device)  # ensure same device
        loss_rep = torch.tensor(0.0, device=loss_nll.device)

        if self.kd_alpha > 0 or self.rep_alpha > 0:
            # For teacher, we only need input_ids and attention_mask for encoder hidden states
            # and decoder_input_ids (derived from labels) for logits and decoder hidden states.
            # The CharacterAwareMTModel's `forward` passes `inputs_embeds` based on `input_ids`.
            # The teacher model will use its own standard embedding for `input_ids`.
            current_teacher_src_nllb = inputs["teacher_src_nllb_code"][0]

            teacher_tokenizer_for_batch = AutoTokenizer.from_pretrained(
                self.teacher.config._name_or_path,  # Get path from teacher model
                src_lang=current_teacher_src_nllb
            )

            teacher_tokenized_inputs = teacher_tokenizer_for_batch(
                inputs["teacher_input_text"],  # List of unprefixed source strings
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SUBWORD_SEQ_LEN  # Use global config
            ).to(self.args.device)

            # If student_tokenizer.pad_token_id is -100 for labels, convert for teacher.
            teacher_labels_for_kd = inputs["labels"].clone()
            if self.tokenizer and hasattr(self.tokenizer, 'pad_token_id'):  # self.tokenizer is student's
                teacher_labels_for_kd[
                    teacher_labels_for_kd == -100] = self.teacher.config.pad_token_id if self.teacher.config.pad_token_id is not None else teacher_tokenizer_for_batch.pad_token_id

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=teacher_tokenized_inputs.input_ids,
                    attention_mask=teacher_tokenized_inputs.attention_mask,
                    labels=teacher_labels_for_kd, # Using student's label IDs, see comment above
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True
                )

            if self.kd_alpha > 0 and hasattr(teacher_outputs, "logits") and hasattr(student_outputs, "logits"):
                student_logits_for_kd = student_outputs.logits
                teacher_logits_for_kd = teacher_outputs.logits

                soft_student_log_probs = F.log_softmax(student_logits_for_kd / self.temperature, dim=-1)
                soft_teacher_probs = F.softmax(teacher_logits_for_kd / self.temperature, dim=-1)
                loss_kd = self.kl_div_loss(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)

                # 2. Patient Representation Distillation
            if self.rep_alpha > 0 and hasattr(teacher_outputs, "encoder_hidden_states") and hasattr(student_outputs,
                                                                                                    "encoder_hidden_states"):
                student_enc_hidden_states = student_outputs.encoder_hidden_states
                teacher_enc_hidden_states = teacher_outputs.encoder_hidden_states
                if student_enc_hidden_states is not None and teacher_enc_hidden_states is not None:
                    current_rep_loss = 0.0
                    num_layers_distilled = 0
                    for layer_idx in self.rep_layers_indices:
                        if abs(layer_idx) < len(student_enc_hidden_states) and \
                                abs(layer_idx) < len(teacher_enc_hidden_states):
                            student_hs = student_enc_hidden_states[layer_idx]
                            teacher_hs = teacher_enc_hidden_states[layer_idx]
                            if student_hs.shape == teacher_hs.shape:
                                current_rep_loss += self.mse_loss(student_hs, teacher_hs)
                                num_layers_distilled += 1
                            else:
                                logger.warning(
                                    f"Skipping hidden state distillation for layer {layer_idx} due to shape mismatch: "
                                    f"Student: {student_hs.shape}, Teacher: {teacher_hs.shape}")
                    if num_layers_distilled > 0:
                        loss_rep = current_rep_loss / num_layers_distilled
                else:
                    logger.warning("Encoder hidden states not available from student or teacher for rep distillation.")

            total_loss = loss_nll + self.kd_alpha * loss_kd + self.rep_alpha * loss_rep
            return (total_loss, student_outputs) if return_outputs else total_loss
