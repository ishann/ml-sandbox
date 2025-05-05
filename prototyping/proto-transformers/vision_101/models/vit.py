import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts an input image into a sequence of patch embeddings for a Vision Transformer.

    This module divides the image into non-overlapping patches using a Conv2D layer,
    flattens each patch, and rearranges the output to match the expected input shape
    for a transformer encoder.

    Args:
        img_size (int): Height/width of the input image (assumes square image)
        patch_size (int): Size of each square patch
        in_channels (int): Number of input channels (e.g. 3 for RGB)
        embed_dim (int): Output dimension for each patch embedding

    Input shape:
        x: Tensor of shape (batch_size, in_channels, img_size, img_size)

    Output shape:
        Tensor of shape (batch_size, num_patches, embed_dim)
    """

    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()

        # A Conv2d layer to split the image into non-overlapping patches
        # and project each patch into a vector of dimension 'embed_dim'
        # Input shape: (B, in_channels, img_size, img_size)
        # Output shape: (B, embed_dim, H', W') where H' = W' = img_size / patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) - input image tensor

        # Applies Conv2d to divide image into patches and embed them
        # Output shape: (B, E, H', W') where H' = H/patch_size, W' = W/patch_size
        x = self.proj(x)

        # Flattens spatial dimensions (H', W') into a single sequence dimension
        # Output shape: (B, E, H'*W') => (B, E, num_patches)
        x = x.flatten(2)

        # Transposes to shape expected by transformer: (B, num_patches, E)
        # Output shape: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        # We're now ready to feed into a transformer where each patch in a sample
        # is represented by a vector of length embed_dim.
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for Vision Transformers.

    Wraps PyTorch's built-in nn.MultiheadAttention to apply self-attention over
    a sequence of embedded tokens.

    Args:
        embed_dim (int): Dimensionality of token embeddings
        num_heads (int): Number of attention heads
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        # PyTorch's built-in multi-head self-attention layer
        # Expects input shape: (seq_len, batch_size, embed_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        # Transpose to (seq_len, batch_size, embed_dim) for MultiheadAttention

        attn_output, _ = self.attn(x, x, x)
        # Perform self-attention using same input for query, key, and value
        # Output shape: (seq_len, batch_size, embed_dim)

        return attn_output.transpose(0, 1)
        # Transpose back to (batch_size, seq_len, embed_dim) to match ViT conventions


class TransformerEncoderBlock(nn.Module):
    """
    A single encoder block used in Vision Transformers.

    Consists of:
    - LayerNorm → Multi-Head Self-Attention → Residual Add
    - LayerNorm → MLP → Residual Add

    Args:
        embed_dim (int): Dimensionality of token embeddings
        num_heads (int): Number of attention heads
        mlp_dim (int): Hidden dimension of the MLP block
        dropout (float): Dropout probability
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()

        # First LayerNorm before attention
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention layer
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)

        # Second LayerNorm before MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward network (MLP) with GELU activation and dropout
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        # Apply attention to normalized input, then add residual connection
        x = x + self.attn(self.norm1(x))

        # Apply MLP to normalized input, then add residual connection
        x = x + self.mlp(self.norm2(x))

        # Output shape: (B, seq_len, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.

    This implementation splits the input image into patches, linearly embeds each patch,
    adds positional encodings and a learnable [CLS] token, then feeds the resulting
    sequence into a stack of Transformer encoder blocks. The output of the [CLS] token
    is used for classification.

    Args:
        img_size (int): Height/width of the input image (assumes square image)
        patch_size (int): Size of each square patch
        in_channels (int): Number of input channels (e.g. 3 for RGB)
        num_classes (int): Number of output classes
        embed_dim (int): Dimension of the token embeddings
        depth (int): Number of transformer encoder layers
        num_heads (int): Number of attention heads per transformer block
        mlp_dim (int): Hidden dimension of the MLP layers in transformer blocks

    Input shape:
        x: Tensor of shape (batch_size, in_channels, img_size, img_size)

    Output shape:
        Tensor of shape (batch_size, num_classes)
    """

    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=6,
        num_heads=8,
        mlp_dim=128,
    ):
        super().__init__()

        # Module to embed image patches into vectors of dimension 'embed_dim'
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Compute total number of patches per image
        num_patches = (img_size // patch_size) ** 2

        # Learnable [CLS] token that represents the entire image for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for each patch + the [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Stack of transformer encoder blocks
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
                for _ in range(depth)
            ]
        )

        # Final layer normalization before classification head
        self.norm = nn.LayerNorm(embed_dim)

        # Linear classification head that maps the CLS token to class logits
        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):

        # Convert input image into a sequence of patch embeddings
        # Shape: (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        batch_size = x.size(0)

        # Duplicate the learnable [CLS] token for each sample in the batch
        # Shape: (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend the [CLS] token to the patch sequence
        # Shape: (B, num_patches + 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding to the sequence
        x = x + self.pos_embed

        # Pass the sequence through transformer encoder blocks
        x = self.encoder(x)

        # Apply layer normalization
        x = self.norm(x)

        # Return classification logits from the [CLS] token output
        # Shape: (B, num_classes)
        return self.head(x[:, 0])
