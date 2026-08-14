"""
V-JEPA Chapters 5, 6, 7 — TTS Audio Generator
Giọng: en-GB-RyanNeural (Male, UK)
Chạy: python visualizations/tts_ch567_full.py
"""
import ssl
import asyncio
import os


def patched_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


ssl.create_default_context = patched_default_context

import edge_tts
from mutagen.mp3 import MP3

VOICE = "en-GB-RyanNeural"
os.makedirs("media", exist_ok=True)

# ── Chapter 5.1 — Loss Function & Stop-Gradient ──────────────────────────
CH5_1 = (
    "The mathematical heart of V-JEPA is its prediction loss function. "
    "We compute the smooth L-1 distance between the predicted latent features, "
    "s-hat-k, and the target features, s-k, averaged over all M masked tokens. "
    "This is a fundamentally different objective from pixel reconstruction — "
    "the loss lives entirely in the abstract semantic latent space. "
    "Now watch the gradient flow during backpropagation. "
    "Gradients propagate backward from the loss through the Predictor "
    "and into the Context Encoder — these two networks actively learn. "
    "But when the gradient signal reaches the Target Encoder boundary, "
    "it hits a mathematical wall: the stop-gradient operator. "
    "This barrier is absolutely essential. "
    "Without it, the model could trivially set both encoders to output "
    "the same constant, making the loss zero without learning anything useful."
)

# ── Chapter 5.2 — Representation Collapse & EMA ──────────────────────────
CH5_2 = (
    "Without proper safeguards, self-supervised learning falls into "
    "a catastrophic failure mode called Representation Collapse. "
    "Imagine every possible video input — a running athlete, a cooking tutorial, "
    "a nature documentary — being mapped to the exact same output vector. "
    "The loss drops to zero, yet the model has learned absolutely nothing. "
    "V-JEPA escapes this trap through two mathematically interlocked mechanisms. "
    "The stop-gradient, which we just saw, cuts the direct path. "
    "The second mechanism is the Exponential Moving Average, or E-M-A. "
    "Instead of training the Target Encoder directly, "
    "its weights are updated as a slow, momentum-weighted average "
    "of the Context Encoder weights. "
    "The momentum m starts at 0.998 and gradually increases to 1.0 over training, "
    "making the target encoder progressively more stable as a fixed reference. "
    "The mathematics here is elegant: when the Predictor achieves optimality under the L-1 loss, "
    "the gradient signal forces the Context Encoder to minimize "
    "the Median Absolute Deviation of the targets. "
    "This means the encoder is mathematically obligated to capture "
    "rich, diverse representations — collapse becomes provably impossible."
)

# ── Chapter 6.1 — Benchmark Comparison ──────────────────────────────────
CH6_1 = (
    "Let us place V-JEPA on the global benchmark leaderboard. "
    "First, Something-Something-v2 — the most demanding test of physical motion understanding. "
    "This dataset is specifically designed so that appearance alone cannot solve it: "
    "you must genuinely understand physical dynamics to succeed. "
    "V-JEPA achieves 71.2 percent accuracy under the fully frozen evaluation protocol. "
    "This is a remarkable 10.0 percentage point improvement over VideoMAE, "
    "11 points over InternVideo, "
    "21 points over DINOv2 — despite DINOv2 training on far more data, "
    "and a stunning 32 points over OpenCLIP. "
    "The gap on Something-Something-v2 is the most telling result, "
    "because large image-language models that have seen billions of images "
    "simply cannot understand physical motion from static appearance cues. "
    "On Kinetics-400, which tests appearance and context recognition, "
    "V-JEPA achieves 82.1 percent, again surpassing all video self-supervised baselines."
)

# ── Chapter 6.2 — Controlled Comparison & Sample Efficiency ──────────────
CH6_2 = (
    "To make the comparison absolutely airtight, "
    "consider the controlled experiment. "
    "We fix the architecture to identical ViT-L 16 models "
    "and train both V-JEPA and VideoMAE exclusively on Kinetics-400. "
    "Under this equal footing, V-JEPA still wins consistently: "
    "plus 0.7 points on Kinetics-400, plus 0.5 points on SSv2, "
    "and an impressive plus 3.4 points on AVA action detection. "
    "But the most striking result is about data efficiency. "
    "V-JEPA achieves all these superior results after seeing "
    "just 210 million training samples. "
    "Compare this to DINOv2, which requires 1,900 million samples — nine times more. "
    "VideoMAEv2 needs 1,600 million. "
    "And OpenCLIP consumes a staggering 39,000 million samples — "
    "nearly 200 times more than V-JEPA. "
    "V-JEPA is not merely more accurate — "
    "it is dramatically more sample-efficient than any competing approach."
)

# ── Chapter 7 — Attentive Probing & Conclusion ───────────────────────────
CH7 = (
    "To measure representation quality with maximum fairness, "
    "V-JEPA employs the frozen evaluation protocol with Attentive Probing. "
    "The entire V-JEPA video encoder — with hundreds of millions of parameters — "
    "is completely locked. Not a single backbone weight is updated during evaluation. "
    "Instead, a lightweight cross-attention probe is trained on top. "
    "A single learnable query token attends across all 1568 spatial-temporal encoder outputs, "
    "intelligently pooling them into one compact task representation. "
    "This pooled vector then feeds a simple linear classifier. "
    "The attentive pooling formula is: "
    "the query attends to keys and values derived from the encoder tokens, "
    "producing a weighted sum that captures task-relevant structure. "
    "Using this single frozen encoder, V-JEPA achieves 77.9 percent on ImageNet-1K "
    "with absolutely no image fine-tuning, "
    "82 percent on Kinetics-400, 71 percent on SSv2, "
    "and leading results across action localization, scene recognition, "
    "and fine-grained species identification. "
    "V-JEPA has proven the ultimate thesis of self-supervised representation learning: "
    "when a machine learns to predict the hidden latent world of video, "
    "it simultaneously acquires the deepest possible understanding "
    "of the entire visual universe."
)

AUDIO_FILES = [
    (CH5_1, "media/ch5_1_new.mp3"),
    (CH5_2, "media/ch5_2_new.mp3"),
    (CH6_1, "media/ch6_1_new.mp3"),
    (CH6_2, "media/ch6_2_new.mp3"),
    (CH7,   "media/ch7_new.mp3"),
]


async def amain():
    print("Generating TTS audio files for V-JEPA Chapters 5, 6, 7...\n")
    for text, path in AUDIO_FILES:
        print(f"  Generating {path} ...")
        await edge_tts.Communicate(text, VOICE).save(path)
        dur = MP3(path).info.length
        words = len(text.split())
        print(f"  OK {path}  [{dur:.1f}s, {words} words]\n")
    print("All audio files generated successfully!")


if __name__ == "__main__":
    asyncio.run(amain())
