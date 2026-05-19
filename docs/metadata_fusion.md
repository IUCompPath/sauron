# Metadata Fusion in AEGIS

This document explains the multi-modal fusion strategy used in the AEGIS MIL models.

## Overview

Medical Imaging tasks often benefit from incorporating clinical metadata (e.g., Patient Age, Sex, Tumor Site, Gene Expressions) alongside the Whole Slide Image (WSI) features. 

The goal of our fusion strategy is to be:
1.  **Additive:** Adding context without destroying image signals.
2.  **Maskable:** The model must function correctly even if metadata is missing (e.g., during inference on a new patient where we only have the slide).
3.  **Principled:** Avoiding ad-hoc operations like "adding vectors" which rely on implicit distribution matching.

## Architecture

We utilize a **Concatenation + Projection** architecture with **Layer Normalization**.

### The Flow
1.  **Image Branch:** The MIL model aggregates tile features into a single Bag Representation (`bag_rep`) of dimension $D$.
2.  **Metadata Branch:** Clinical variables are encoded (one-hot or normalized) and passed through a non-linear encoder (MLP) to produce `encoded_meta` of dimension $M$.
3.  **Fusion:** 
    $$ \text{Fused} = \text{LayerNorm}(\text{ReLU}(\text{Linear}(\text{Concat}(\text{bag\_rep}, \text{encoded\_meta})))) $$ 

### Why not just add them? (`Image + Meta`)
Previous versions used simple addition. This is suboptimal because:
*   It forces the metadata encoder to match the scale and distribution of the image features exactly.
*   If the metadata signal is strong, it can numerically "drown out" subtle image patterns.
*   It assumes the latent space of images and metadata is identical.

### Why not append as a token? (`[Patch1, Patch2, ..., Meta]`)
Appending metadata as just another patch in the bag (for Attention MIL) is efficient but semantically inconsistent. It forces the attention mechanism to treat a vector of clinical data exactly like a vector of image features ("Apples and Oranges"), which restricts the model's ability to process them distinctly.

## Handling Missing Data (Maskability)

A key requirement is robustness. In clinical practice, metadata might be unavailable.

*   **Training:** We recommend using **Modality Dropout** (randomly zeroing out metadata) during training to force the model to learn from images alone.
*   **Inference:** If `metadata` is `None`, the `BaseMILModel` automatically substitutes a **Zero Vector**.
*   **Stability:** The `LayerNorm` in the fusion block is critical here. It ensures that switching from "Metadata Present" to "Metadata Zero" does not cause a massive shift in the magnitude of the features entering the final classifier.

## Usage

All models inheriting from `BaseMILModel` support this out of the box.

```python
# In your model definition
model = MyMILModel(..., metadata_dim=10, metadata_fusion_dim=128)

# Forward pass with data
logits, ... = model(images, metadata=clinical_data)

# Forward pass without data (simulating missing info)
logits, ... = model(images, metadata=None)
```

```