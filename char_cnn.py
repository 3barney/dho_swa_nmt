import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

CHAR_CNN_KERNEL_SIZES = [2, 3, 4]
CHAR_CNN_NUM_FILTERS = 64

# Character level encoder
class CharCNN(nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 char_emb_dim: int,
                 output_dim: int,  # Dimension of the feature vector for each subword
                 kernel_sizes: List[int] = CHAR_CNN_KERNEL_SIZES,
                 num_filters_per_kernel: int = CHAR_CNN_NUM_FILTERS,
                 char_pad_idx: int = 0):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=char_pad_idx)

        # Using Conv1d as it's more natural for sequences of characters
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_emb_dim,
                      out_channels=num_filters_per_kernel,
                      kernel_size=k,
                      padding=(k // 2) if k % 2 == 1 else ((k // 2) - 1, k // 2)  # Maintain length approx.
                      ) for k in kernel_sizes
        ])
        # The output dimension from concatenated conv layers
        concatenated_dim = num_filters_per_kernel * len(kernel_sizes)
        self.fc = nn.Linear(concatenated_dim, output_dim)
        self.dropout = nn.Dropout(0.25)  # Added dropout

    def forward(self, char_ids_batch: torch.Tensor) -> torch.Tensor:
        """
        char_ids_batch: (batch_size, max_subwords_in_batch, max_chars_per_subword)
        Output: (batch_size, max_subwords_in_batch, output_dim)
        """
        batch_size, max_subwords, max_chars = char_ids_batch.shape

        # Reshape for character embedding and convolution: treat each subword as an item in a batch
        # (batch_size * max_subwords, max_chars)
        x = char_ids_batch.view(-1, max_chars)

        # Pass through character embedding layer
        # (batch_size * max_subwords, max_chars, char_emb_dim)
        x = self.char_emb(x)

        # Transpose for Conv1d: (batch_size * max_subwords, char_emb_dim, max_chars)
        x = x.transpose(1, 2)

        # Apply convolutions, activations, and pooling
        conv_outputs = []
        for conv_layer in self.convs:
            conv_out = F.relu(conv_layer(x))
            # Max-over-time pooling: (batch_size * max_subwords, num_filters_per_kernel, 1)
            # The kernel_size for max_pool1d should be the length of the sequence after convolution
            pooled_out = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled_out)

        # Concatenate outputs from different kernel sizes
        # (batch_size * max_subwords, num_filters_per_kernel * len(kernel_sizes))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)

        # Pass through fully connected layer
        # (batch_size * max_subwords, output_dim)
        x = self.fc(x)

        # Reshape back to (batch_size, max_subwords, output_dim)
        output_char_embeddings = x.view(batch_size, max_subwords, -1)
        return output_char_embeddings