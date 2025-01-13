import torch
import torch.nn as nn
import torch.nn.functional as F

class BPPEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(BPPEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, embed_dim))  # Assuming max 500 items
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] where 1 indicates valid tokens
        Returns:
            Encoder outputs: [seq_len, batch_size, embed_dim]
        """
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, embed_dim]
        src_key_padding_mask = ~mask  # Transformer expects mask where True indicates padding
        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

class BPPDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(BPPDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, embed_dim))  # Assuming max 500 items
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: [batch_size, seq_len, output_dim]
            memory: [seq_len, batch_size, embed_dim]
        Returns:
            Output predictions: [batch_size, seq_len, output_dim]
        """
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        tgt = tgt.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        output = self.output_layer(output)  # [batch_size, seq_len, output_dim]
        return output

class BPPModel(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, embed_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        """
        Args:
            input_dim: Dimension of input features (dimensions + rotations = 3 + 3 = 6)
            output_dim: Dimension of output features (positions + rotations = 3 + 3 = 6)
        """
        super(BPPModel, self).__init__()
        self.encoder = BPPEncoder(input_dim, embed_dim, num_heads, num_layers, dropout)
        self.decoder = BPPDecoder(output_dim, embed_dim, num_heads, num_layers, dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: [batch_size, src_seq_len, input_dim]
            tgt: [batch_size, tgt_seq_len, output_dim]
            src_mask: [batch_size, src_seq_len]
            tgt_mask: [batch_size, tgt_seq_len]
        Returns:
            Outputs: [batch_size, tgt_seq_len, output_dim]
        """
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory)
        return output