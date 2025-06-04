import torch
import torch.nn as nn
import logging

from typing import Optional
from transformers import (
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM
)

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
                 freeze_base_model: bool = True,
                 use_bnb_quantization: bool = False):
        super().__init__()
        logger.info(f"Initializing CharacterAwareMTModel with base: {base_model_name_or_path}")

        if use_bnb_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # On Ampere GPUs or newer
                bnb_4bit_use_double_quant=True,
            )
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto"  # Automatically places parts of model on devices, good for large models
            )
            logger.info("Base model loaded with 4-bit BNB quantization.")
        else:
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_or_path)
            logger.info("Base model loaded without BNB quantization.")

        self.config = self.base_model.config
        base_model_hidden_size = self.config.d_model  # Standard attribute for hidden size

        # Determine CharCNN output dimension
        # If char_cnn_output_dim_ratio is 0, it means we might not use char_cnn or use a different combination strategy.
        # For now, we assume if it's >0, we use it.
        self.char_cnn_output_dim = int(
            base_model_hidden_size * char_cnn_output_dim_ratio) if char_cnn_output_dim_ratio > 0 else 0

        if self.char_cnn_output_dim > 0:
            self.char_cnn = CharCNN(
                char_vocab_size=char_vocab_size,
                char_emb_dim=char_emb_dim,
                output_dim=self.char_cnn_output_dim,
                char_pad_idx=char_pad_idx
            )
            logger.info(f"CharCNN initialized with output_dim: {self.char_cnn_output_dim}")
            # Projection layer to combine subword and char embeddings back to base_model's expected hidden size
            self.projection_layer = nn.Linear(
                base_model_hidden_size + self.char_cnn_output_dim,
                base_model_hidden_size
            )
            logger.info("Projection layer initialized for combined embeddings.")
        else:
            self.char_cnn = None
            self.projection_layer = None
            logger.info("CharCNN is disabled as char_cnn_output_dim_ratio is 0 or less.")

        if freeze_base_model:
            for param_name, param in self.base_model.named_parameters():
                # Important: We usually want to fine-tune the embedding layer of the base model
                # if we are combining its output with char embeddings.
                # For this setup, where we take NLLB's subword embeddings and *add/concat* char features,
                # we MIGHT want the subword embeddings to also adapt.
                # However, for strict distillation and resource constraints, freezing all of NLLB
                # including its original input embeddings can be a starting point.
                # The CharacterAwareModel in the original script implicitly made base embeddings trainable
                # by just adding to them.
                # If we *replace* input_ids with inputs_embeds, the original embedding layer of base_model
                # isn't directly used in the forward pass of the base_model itself for those embeddings.
                # Let's freeze all base model parameters for now.
                param.requires_grad = False
            logger.info("Froze all parameters of the base model.")

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                source_char_ids: Optional[torch.Tensor] = None,  # (batch, seq_len, char_len)
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):

        # Get subword embeddings from the base model's embedding layer
        # Even if the base_model is frozen, we can still use its embedding layer separately
        # if we want its original subword representations.
        subword_embeddings = self.base_model.get_input_embeddings()(input_ids)

        if self.char_cnn and source_char_ids is not None and self.projection_layer:
            # Get character-based embeddings for each subword
            # char_cnn_output: (batch_size, max_subwords_in_batch, char_cnn_output_dim)
            char_cnn_output = self.char_cnn(source_char_ids)

            # Concatenate subword and character embeddings
            # subword_embeddings: (batch_size, max_subwords_in_batch, base_model_hidden_size)
            combined_embeddings = torch.cat((subword_embeddings, char_cnn_output), dim=-1)

            # Project the combined embeddings to match the base model's expected hidden size
            inputs_embeds = self.projection_layer(combined_embeddings)
        else:  # If CharCNN is disabled or source_char_ids not provided
            inputs_embeds = subword_embeddings

        # Pass inputs_embeds to the base model
        # The `input_ids` argument to base_model is effectively ignored if `inputs_embeds` is provided.
        # However, decoder_input_ids are typically derived from labels by the model internally,
        # or by DataCollatorForSeq2Seq. We need to ensure this works.
        # If labels are provided, HF models usually create decoder_input_ids by shifting labels.
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
