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
        student_outputs = model(**inputs, output_hidden_states=True, output_attentions=False, return_dict=True)
        loss_nll = student_outputs.loss

        loss_kd = 0.0
        loss_rep = 0.0

        if self.kd_alpha > 0 or self.rep_alpha > 0:
            # For teacher, we only need input_ids and attention_mask for encoder hidden states
            # and decoder_input_ids (derived from labels) for logits and decoder hidden states.
            # The CharacterAwareMTModel's `forward` passes `inputs_embeds` based on `input_ids`.
            # The teacher model will use its own standard embedding for `input_ids`.
            teacher_input_ids = inputs.get("input_ids")
            teacher_attention_mask = inputs.get("attention_mask")
            teacher_labels = inputs.get("labels")

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    labels=teacher_labels,  # Teacher also needs labels to compute its logits for KD
                    output_hidden_states=True,
                    output_attentions=False,  # Not needed for distillation usually
                    return_dict=True
                )

            # 1. Knowledge Distillation on Logits (KD)
            if self.kd_alpha > 0:
                student_logits_for_kd = student_outputs.logits
                teacher_logits_for_kd = teacher_outputs.logits

                # Soften probabilities with temperature
                soft_student_log_probs = F.log_softmax(student_logits_for_kd / self.temperature, dim=-1)
                soft_teacher_probs = F.softmax(teacher_logits_for_kd / self.temperature, dim=-1)

                loss_kd = self.kl_div_loss(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)
                # Multiply by T^2 to scale gradients appropriately (Hinton et al., 2015)

            # 2. Patient Representation Distillation (Hidden States MSE)
            if self.rep_alpha > 0:
                # Distill from encoder hidden states. Decoder hidden states can also be used.
                # Ensure `output_hidden_states=True` for both student and teacher.
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
                            # If dimensions don't match, a projection might be needed here for student_hs
                            # This assumes student and teacher encoder layers have same hidden dim,
                            # which is often true when distilling between similar family models or using projection in student.
                            if student_hs.shape == teacher_hs.shape:
                                current_rep_loss += self.mse_loss(student_hs, teacher_hs)
                                num_layers_distilled += 1
                            else:
                                logger.warning(
                                    f"Skipping hidden state distillation for layer {layer_idx} due to shape mismatch: "
                                    f"Student: {student_hs.shape}, Teacher: {teacher_hs.shape}")
                    if num_layers_distilled > 0:
                        loss_rep = current_rep_loss / num_layers_distilled  # Average MSE across selected layers
                else:
                    logger.warning("No valid layers found for hidden state representation distillation.")

        # Total loss
        total_loss = (1 - self.kd_alpha - self.rep_alpha) * loss_nll + \
                     self.kd_alpha * loss_kd + \
                     self.rep_alpha * loss_rep

        # Ensure alphas sum to 1 or less, or adjust logic for NLL weight.
        # The above assumes NLL weight is (1 - kd_alpha - rep_alpha).
        # A more common formulation might be:
        # loss = (1 - alpha_total) * loss_nll + alpha_kd * loss_kd + alpha_rep * loss_rep
        # Or simply: loss = loss_nll + weight_kd * loss_kd + weight_rep * loss_rep
        # For this implementation, the original meaning from your script was:
        # loss = loss_ce + self.kd_alpha * loss_kd + self.rep_alpha * loss_rep
        # Let's stick to this, where NLL is the primary loss and others are additive weighted terms.
        total_loss = loss_nll + self.kd_alpha * loss_kd + self.rep_alpha * loss_rep

        return (total_loss, student_outputs) if return_outputs else total_loss