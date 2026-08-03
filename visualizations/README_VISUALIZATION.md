# V-JEPA Manim Animation

Manim Community Edition visualization of the V-JEPA (Video Joint Embedding Predictive Architecture) pipeline, ICLR 2024.

## Setup

### 1. Install Manim Community Edition

```bash
pip install manim
```

Requires Python 3.9+ and `manim>=0.18.0`.

### 2. Verify

```bash
manim --version
```

## Rendering

All scenes are defined in `visualizations/vjepa_manim.py`.

### Individual scenes

| Scene Class | Description |
|---|---|
| `TitleScene` | Title card & pixel-reconstruction vs latent-prediction contrast |
| `PatchTokenizeScene` | 3D video clip → 3D convolution → 1568 token sequence |
| `MultiBlockMaskScene` | Short-range (8 blocks, 0.15 scale) & long-range (2 blocks, 0.7 scale) masks |
| `ArchitectureScene` | Context Encoder, Target Encoder (EMA), Predictor, L₁ loss |
| `HighlightsScene` | Key results & downstream benchmarks (ViT-L/16) |

**Preview quality (fast):**

```bash
manim -pql visualizations/vjepa_manim.py TitleScene
manim -pql visualizations/vjepa_manim.py PatchTokenizeScene
manim -pql visualizations/vjepa_manim.py MultiBlockMaskScene
manim -pql visualizations/vjepa_manim.py ArchitectureScene
manim -pql visualizations/vjepa_manim.py HighlightsScene
```

**High quality:**

```bash
manim -pqh visualizations/vjepa_manim.py <SceneName>
```

### All-in-one composite scene

```bash
manim -pql visualizations/vjepa_manim.py VJEPAAllScenes
```

Renders all 5 sections sequentially in a single video (uses `self.next_section()` for chapter markers).

## Output

Rendered `.mp4` files appear in `media/videos/vjepa_manim/`.

For the all-in-one scene, sections can be navigated via chapter markers in players like mpv or VLC.

## Architecture Mapping to Codebase

| Component | Source File |
|---|---|
| 3D Patch Embedding (Conv2×16×16) | `src/models/utils/patch_embed.py` — `PatchEmbed3D` |
| Vision Transformer backbone | `src/models/vision_transformer.py` — `VisionTransformer` |
| 3D Multi-Block Masking | `src/masks/multiblock3d.py` — `MaskCollator`, `_MaskGenerator` |
| Context/Target Encoder | `src/models/vision_transformer.py` (shared architecture) |
| Predictor (narrow ViT) | `src/models/predictor.py` — `VisionTransformerPredictor` |
| Multi-Mask Wrapper | `src/models/utils/multimask.py` |
| Attentive Pooler (eval) | `src/models/attentive_pooler.py` |
| Training loop (EMA, loss) | `app/vjepa/train.py` |
| Config (mask params, etc.) | `configs/pretrain/vitl16.yaml` |

## Key Dimensions (from implementation)

| Parameter | Value |
|---|---|
| Input clip | 16 × 224 × 224 × 3 |
| Tubelet size (temporal patch) | 2 frames |
| Spatial patch size | 16 × 16 |
| 3D Conv kernel/stride | 2 × 16 × 16 |
| Patch grid | 8T × 14H × 14W = 1568 tokens |
| Embedding dim (ViT-L) | d = 1024 |
| Predictor dim | 384 (narrow) |
| Predictor depth | 12 |
| Short-range mask | 8 blocks, spatial scale 0.15 |
| Long-range mask | 2 blocks, spatial scale 0.7 |
| EMA momentum range | [0.998, 1.0] |
| Loss | L₁ (mean absolute error in latent space) |
