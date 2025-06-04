import torch
import torch.nn as nn
import logging

from typing import List, Optional
from transformers import (
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from char_cnn import CharCNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHAR_EMB_DIM = 64  # Reduced for memory

class CharacterAwareMTModel(nn.Module):
    def __init__(self,
                 base_model_name_or_path: str,
                 char_vocab_size: int,
                 char_emb_dim: int = CHAR_EMB_DIM,
                 char_cnn_output_dim_ratio: float = 0.25,  # Ratio of base_model hidden_size for char_cnn output
                 char_pad_idx: int = 0,
                 # freeze_base_model: bool = True,
                 use_bnb_quantization_for_base_before_peft: bool = False,
                 use_bnb_quantization: bool = False,
                 peft_lora_r: int = 8,
                 peft_lora_alpha: int = 16,
                 peft_lora_dropout: float = 0.05,
                 peft_target_modules: Optional[List[str]] = None
                 ):
        super().__init__()
        logger.info(f"Initializing CharacterAwareMTModel with PEFT (LoRA) for base: {base_model_name_or_path}")

        quantization_config_for_base = None
        if use_bnb_quantization_for_base_before_peft:
            quantization_config_for_base = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit BNB quantization for the base model before applying PEFT (QLoRA style).")

        # Load the base model
        base_model_raw = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=quantization_config_for_base,  # Apply if QLoRA style
            device_map="auto" if quantization_config_for_base else None  # device_map useful for quantized models
        )

        if quantization_config_for_base:
            # Prepares the quantized model for k-bit training (important for QLoRA)
            base_model_raw = prepare_model_for_kbit_training(base_model_raw)
            logger.info("Base model prepared for k-bit training (QLoRA).")

            # TODO: Migratr this to use nllb model
            # Define LoRA configuration
            # For T5/mT5 models, common target_modules are 'q' and 'v' in SelfAttention and EncDecAttention
            # If None, PEFT might try to infer, or you can list them explicitly if known
            # e.g. ["q_proj", "v_proj"] or for T5 more like ["q", "v"] in SelfAttention and EncDecAttention
            # A common approach for T5 models is to target "q" and "v" across all attention blocks.
            # To be precise, you'd typically list modules like:
            # 'encoder.block.0.layer.0.SelfAttention.q', 'encoder.block.0.layer.0.SelfAttention.v',
            # 'encoder.block.0.layer.1.EncDecAttention.q', 'encoder.block.0.layer.1.EncDecAttention.v',
            # ... and similar for the decoder.
            # However, simply passing ["q", "v"] often works if the library can map them correctly for the architecture.
            # For a more robust way or if ["q","v"] is too general, check model's named_modules().
            # For mT5, ["q", "v"] is a common shorthand for query and value projection layers in attention.
            if peft_target_modules is None:
                # Heuristic for T5-like models based on common practice.
                # Inspect your specific STUDENT_CKPT's named_modules to be certain.
                peft_target_modules = ["q", "v"]
                logger.info(f"Using default PEFT target modules for T5-like model: {peft_target_modules}")

            lora_config = LoraConfig(
                r=peft_lora_r,
                lora_alpha=peft_lora_alpha,
                target_modules=peft_target_modules,
                lora_dropout=peft_lora_dropout,
                bias="none",  # Typically 'none' for LoRA on linear layers
                task_type=TaskType.SEQ_2_SEQ_LM,
                # modules_to_save = ["lm_head", "char_cnn", "projection_layer"] # If you want to also treat these as part of PEFT save/load
                # However, CharCNN and projection_layer are custom.
                # It's often easier to handle them as separate, fully trained parts.
                # If lm_head needs training and is part of base, add it here.
            )

            # Apply PEFT to the base model
            self.base_model = get_peft_model(base_model_raw, lora_config)
            logger.info("PEFT (LoRA) applied to the base model.")
            self.base_model.print_trainable_parameters()

            self.config = self.base_model.config  # Get config from PEFT model (it forwards to base)
            base_model_hidden_size = self.config.d_model

            self.char_cnn_output_dim = int(
                base_model_hidden_size * char_cnn_output_dim_ratio) if char_cnn_output_dim_ratio > 0 else 0

            if self.char_cnn_output_dim > 0:
                self.char_cnn = CharCNN(
                    char_vocab_size=char_vocab_size,
                    char_emb_dim=char_emb_dim,
                    output_dim=self.char_cnn_output_dim,
                    char_pad_idx=char_pad_idx,
                    kernel_sizes=CharCNN.CHAR_CNN_KERNEL_SIZES,  # Using global config
                    num_filters_per_kernel=CharCNN.CHAR_CNN_NUM_FILTERS_PER_KERNEL  # Using global config
                )
                logger.info(f"CharCNN initialized with output_dim: {self.char_cnn_output_dim}")
                self.projection_layer = nn.Linear(
                    base_model_hidden_size + self.char_cnn_output_dim,
                    base_model_hidden_size
                )
                logger.info("Projection layer initialized for combined embeddings.")
            else:
                self.char_cnn = None
                self.projection_layer = None
                logger.info("CharCNN is disabled as char_cnn_output_dim_ratio is 0 or less.")

            # Note: The `freeze_base_model` and `freeze_base_embeddings` parameters
            # are implicitly handled by PEFT (which freezes most of the base model)
            # and how you define `lora_config.target_modules` and `lora_config.modules_to_save`.
            # The CharCNN and projection_layer are n

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                source_char_ids: Optional[torch.Tensor] = None,  # (batch, seq_len, char_len)
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                # Add any other arguments the base model's forward might expect for seq2seq
                decoder_input_ids: Optional[torch.Tensor] = None  # Often handled by HF Trainer/DataCollator
                ):

        # Get subword embeddings from the base model's (PeftModel's) embedding layer
        # The PeftModel will correctly delegate to the base model's actual embedding layer.
        # If PEFT is applied to embedding layers, it will be handled.
        subword_embeddings = self.base_model.get_input_embeddings()(input_ids)

        if self.char_cnn and source_char_ids is not None and self.projection_layer:
            char_cnn_output = self.char_cnn(source_char_ids)
            combined_embeddings = torch.cat((subword_embeddings, char_cnn_output), dim=-1)
            inputs_embeds = self.projection_layer(combined_embeddings)
        else:
            inputs_embeds = subword_embeddings

        # Pass inputs_embeds to the base_model (which is now a PeftModel)
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,  # Pass if provided/needed
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
