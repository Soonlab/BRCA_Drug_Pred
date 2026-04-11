"""
PathOmicDRP: Pathology-Omics Drug Response Predictor

Multi-modal deep learning framework integrating:
  - Genomic features (mutation binary + TMB)
  - Transcriptomic features (pathway-level scores)
  - Proteomic features (RPPA protein expression)
  - Histopathology features (UNI foundation model + ABMIL)

Fusion: Cross-attention between omics tokens and histology tokens
Output: Drug response prediction (IC50 regression / binary classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. Modality-Specific Encoders
# ---------------------------------------------------------------------------

class GenomicEncoder(nn.Module):
    """Encode binary mutation matrix + TMB into genomic tokens."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_tokens: int = 8, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Learnable token assignment: project to n_tokens
        self.token_proj = nn.Linear(hidden_dim, n_tokens)
        self.token_embed = nn.Linear(1, hidden_dim)
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) binary mutation + TMB features
        Returns:
            tokens: (B, n_tokens, hidden_dim) genomic tokens
        """
        h = self.projection(x)  # (B, hidden_dim)
        # Create tokens via learned decomposition
        weights = torch.softmax(self.token_proj(h), dim=-1)  # (B, n_tokens)
        tokens = weights.unsqueeze(-1) * h.unsqueeze(1)  # (B, n_tokens, hidden_dim)
        return tokens


class PathwayTokenizer(nn.Module):
    """Tokenize transcriptomic features at pathway level.

    Each pathway becomes a token, enabling biologically meaningful
    cross-attention with histology patches (SurvPath-inspired).
    """

    def __init__(self, n_pathways: int, genes_per_pathway: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_pathways = n_pathways
        self.pathway_encoder = nn.Sequential(
            nn.Linear(genes_per_pathway, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_pathways, genes_per_pathway) pathway-grouped expression
               OR (B, n_pathways) if already pathway scores (ssGSEA)
        Returns:
            tokens: (B, n_pathways, hidden_dim) pathway tokens
        """
        if x.dim() == 2:
            # Pathway scores: expand to (B, n_pathways, 1) then project
            x = x.unsqueeze(-1)
        tokens = self.pathway_encoder(x)  # (B, n_pathways, hidden_dim)
        tokens = self.layer_norm(tokens)
        return tokens


class ProteomicEncoder(nn.Module):
    """Encode RPPA protein/phosphoprotein expression into protein tokens."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_tokens: int = 16, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_proj = nn.Linear(hidden_dim, n_tokens)
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) RPPA protein expression
        Returns:
            tokens: (B, n_tokens, hidden_dim) protein tokens
        """
        h = self.projection(x)  # (B, hidden_dim)
        weights = torch.softmax(self.token_proj(h), dim=-1)  # (B, n_tokens)
        tokens = weights.unsqueeze(-1) * h.unsqueeze(1)  # (B, n_tokens, hidden_dim)
        return tokens


# ---------------------------------------------------------------------------
# 2. Histopathology Branch (ABMIL)
# ---------------------------------------------------------------------------

class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning for WSI aggregation.

    Takes pre-extracted patch features (e.g., from UNI) and produces
    a slide-level embedding via gated attention.
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256, dropout: float = 0.1, n_tokens: int = 1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim // 2, n_tokens)
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (B, N_patches, feature_dim) patch features from UNI
            mask: (B, N_patches) boolean mask for valid patches
        Returns:
            slide_tokens: (B, n_tokens, hidden_dim) slide-level token(s)
            attention_weights: (B, n_tokens, N_patches) for interpretability
        """
        h = self.feature_proj(x)  # (B, N, hidden_dim)

        a_V = self.attention_V(h)  # (B, N, hidden_dim//2)
        a_U = self.attention_U(h)  # (B, N, hidden_dim//2)
        a = self.attention_W(a_V * a_U)  # (B, N, n_tokens)

        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        a = a.transpose(1, 2)  # (B, n_tokens, N)
        attention_weights = F.softmax(a, dim=-1)  # (B, n_tokens, N)

        slide_tokens = torch.bmm(attention_weights, h)  # (B, n_tokens, hidden_dim)
        return slide_tokens, attention_weights


# ---------------------------------------------------------------------------
# 3. Cross-Attention Fusion Module
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between two sets of tokens."""

    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_1to2 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_2to1 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1a = nn.LayerNorm(hidden_dim)
        self.norm1b = nn.LayerNorm(hidden_dim)
        self.norm2a = nn.LayerNorm(hidden_dim)
        self.norm2b = nn.LayerNorm(hidden_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens1: torch.Tensor, tokens2: torch.Tensor) -> tuple:
        """
        Args:
            tokens1: (B, N1, D) e.g., omics tokens
            tokens2: (B, N2, D) e.g., histology tokens
        Returns:
            updated_tokens1, updated_tokens2
        """
        # tokens1 attends to tokens2
        attn_out1, _ = self.cross_attn_1to2(
            query=tokens1, key=tokens2, value=tokens2
        )
        tokens1 = self.norm1a(tokens1 + attn_out1)
        tokens1 = self.norm1b(tokens1 + self.ffn1(tokens1))

        # tokens2 attends to tokens1
        attn_out2, _ = self.cross_attn_2to1(
            query=tokens2, key=tokens1, value=tokens1
        )
        tokens2 = self.norm2a(tokens2 + attn_out2)
        tokens2 = self.norm2b(tokens2 + self.ffn2(tokens2))

        return tokens1, tokens2


class MultiModalFusion(nn.Module):
    """Fuse omics tokens and histology tokens via cross-attention layers."""

    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # Self-attention over all tokens after cross-attention
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )

    def forward(self, omics_tokens: torch.Tensor, histo_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            omics_tokens: (B, N_omics, D) concatenated genomic + pathway + protein tokens
            histo_tokens: (B, N_histo, D) histology tokens (optional, None if missing)
        Returns:
            fused: (B, N_total, D) fused representation
        """
        if histo_tokens is not None:
            for layer in self.layers:
                omics_tokens, histo_tokens = layer(omics_tokens, histo_tokens)
            all_tokens = torch.cat([omics_tokens, histo_tokens], dim=1)
        else:
            all_tokens = omics_tokens

        fused = self.self_attn(all_tokens)
        return fused


# ---------------------------------------------------------------------------
# 4. Prediction Head
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """Drug response prediction from fused multi-modal representation."""

    def __init__(self, hidden_dim: int = 256, n_drugs: int = 1, task: str = 'regression', dropout: float = 0.2):
        super().__init__()
        self.task = task
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_drugs),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_tokens: (B, N, D) fused multi-modal tokens
        Returns:
            pred: (B, n_drugs) predicted IC50 or response probability
        """
        # Attention-weighted pooling over tokens
        attn_scores = self.pool_attn(fused_tokens).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N)
        pooled = torch.bmm(attn_weights.unsqueeze(1), fused_tokens).squeeze(1)  # (B, D)

        pred = self.head(pooled)  # (B, n_drugs)
        if self.task == 'classification':
            pred = torch.sigmoid(pred)
        return pred


# ---------------------------------------------------------------------------
# 5. Full Model: PathOmicDRP
# ---------------------------------------------------------------------------

class PathOmicDRP(nn.Module):
    """
    PathOmicDRP: Multi-modal drug response prediction model.

    Integrates genomic (mutation), transcriptomic (pathway scores),
    proteomic (RPPA), and histopathology (UNI features) through
    cross-attention fusion.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.1)

        # --- Modality Encoders ---
        self.genomic_encoder = GenomicEncoder(
            input_dim=config['genomic_dim'],
            hidden_dim=hidden_dim,
            n_tokens=config.get('genomic_tokens', 8),
            dropout=dropout,
        )
        self.pathway_tokenizer = PathwayTokenizer(
            n_pathways=config['n_pathways'],
            genes_per_pathway=1,  # ssGSEA scores: 1 value per pathway
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.proteomic_encoder = ProteomicEncoder(
            input_dim=config['proteomic_dim'],
            hidden_dim=hidden_dim,
            n_tokens=config.get('proteomic_tokens', 16),
            dropout=dropout,
        )

        # Histology branch (optional — used when H&E features available)
        self.use_histology = config.get('use_histology', False)
        if self.use_histology:
            self.histology_encoder = ABMIL(
                feature_dim=config.get('histo_feature_dim', 1024),
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_tokens=config.get('histo_tokens', 16),
            )

        # --- Fusion ---
        self.fusion = MultiModalFusion(
            hidden_dim=hidden_dim,
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_fusion_layers', 2),
            dropout=dropout,
        )

        # --- Prediction Head ---
        self.prediction_head = PredictionHead(
            hidden_dim=hidden_dim,
            n_drugs=config.get('n_drugs', 1),
            task=config.get('task', 'regression'),
            dropout=config.get('head_dropout', 0.2),
        )

        # --- Modality dropout for robustness ---
        self.modality_dropout_p = config.get('modality_dropout', 0.0)

    def _modality_dropout(self, tokens: torch.Tensor, name: str) -> torch.Tensor:
        """Randomly zero out entire modality tokens during training."""
        if self.training and self.modality_dropout_p > 0:
            if torch.rand(1).item() < self.modality_dropout_p:
                return torch.zeros_like(tokens)
        return tokens

    def forward(
        self,
        genomic: torch.Tensor,
        transcriptomic: torch.Tensor,
        proteomic: torch.Tensor,
        histology: torch.Tensor = None,
        histo_mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            genomic: (B, genomic_dim) mutation features
            transcriptomic: (B, n_pathways) pathway scores
            proteomic: (B, proteomic_dim) RPPA features
            histology: (B, N_patches, histo_feature_dim) UNI patch features (optional)
            histo_mask: (B, N_patches) valid patch mask (optional)
        Returns:
            dict with 'prediction', and optionally 'attention_weights'
        """
        # Encode each modality into tokens
        gen_tokens = self._modality_dropout(
            self.genomic_encoder(genomic), 'genomic'
        )
        path_tokens = self._modality_dropout(
            self.pathway_tokenizer(transcriptomic), 'transcriptomic'
        )
        prot_tokens = self._modality_dropout(
            self.proteomic_encoder(proteomic), 'proteomic'
        )

        # Concatenate omics tokens
        omics_tokens = torch.cat([gen_tokens, path_tokens, prot_tokens], dim=1)

        # Histology branch
        histo_tokens = None
        histo_attn = None
        if self.use_histology and histology is not None:
            histo_tokens, histo_attn = self.histology_encoder(histology, histo_mask)
            histo_tokens = self._modality_dropout(histo_tokens, 'histology')

        # Cross-attention fusion
        fused = self.fusion(omics_tokens, histo_tokens)

        # Prediction
        prediction = self.prediction_head(fused)

        output = {'prediction': prediction}
        if histo_attn is not None:
            output['histo_attention'] = histo_attn

        return output


# ---------------------------------------------------------------------------
# 6. Configuration helpers
# ---------------------------------------------------------------------------

def get_default_config(
    genomic_dim: int = 193,
    n_pathways: int = 50,
    proteomic_dim: int = 464,
    n_drugs: int = 1,
    use_histology: bool = False,
) -> dict:
    """Return default model configuration."""
    return {
        'genomic_dim': genomic_dim,
        'n_pathways': n_pathways,
        'proteomic_dim': proteomic_dim,
        'hidden_dim': 256,
        'dropout': 0.1,
        'head_dropout': 0.2,
        'genomic_tokens': 8,
        'proteomic_tokens': 16,
        'histo_tokens': 16,
        'histo_feature_dim': 1024,  # UNI ViT-L output dim
        'n_heads': 8,
        'n_fusion_layers': 2,
        'n_drugs': n_drugs,
        'task': 'regression',
        'use_histology': use_histology,
        'modality_dropout': 0.15,
    }


if __name__ == '__main__':
    # Quick test
    config = get_default_config(use_histology=True)
    model = PathOmicDRP(config)

    B = 4
    genomic = torch.randn(B, 193)
    transcriptomic = torch.randn(B, 50)
    proteomic = torch.randn(B, 464)
    histology = torch.randn(B, 100, 1024)  # 100 patches, UNI features

    output = model(genomic, transcriptomic, proteomic, histology)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prediction shape: {output['prediction'].shape}")
    print(f"Histo attention shape: {output['histo_attention'].shape}")

    # Without histology
    config_no_histo = get_default_config(use_histology=False)
    model2 = PathOmicDRP(config_no_histo)
    output2 = model2(genomic, transcriptomic, proteomic)
    print(f"\nOmics-only parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print(f"Omics-only prediction: {output2['prediction'].shape}")
