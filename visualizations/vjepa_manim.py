# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Manim Community Edition animation script for V-JEPA (Video Joint Embedding
Predictive Architecture) — ICLR 2024.

Scene flow:
  1. TitleScene          — title card, pixel-reconstruction vs latent prediction contrast
  2. PatchTokenizeScene   — 3D video clip → 3D convolution → 1568 token sequence
  3. MultiBlockMaskScene  — short-range and long-range 3D multi-block masks
  4. ArchitectureScene    — context branch, target branch (EMA), predictor branch, L₁ loss
  5. HighlightsScene      — key results and downstream performance summary

Usage:
    manim -pql visualizations/vjepa_manim.py TitleScene
    manim -pqh visualizations/vjepa_manim.py VJEPAAllScenes
"""

from manim import *

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_PROBLEM   = "#e74c3c"   # red
C_SOLUTION  = "#2ecc71"   # green
C_INPUT     = "#3498db"   # blue
C_TOKEN     = "#9b59b6"   # purple
C_MASK_BG   = "#34495e"   # dark slate
C_MASK_KEEP = "#2ecc71"   # green (kept tokens)
C_MASK_PRED = "#e74c3c"   # red (masked / predicted)
C_CTXT_ENC  = "#f39c12"   # orange (context encoder)
C_PREDICTOR = "#1abc9c"   # teal (predictor)
C_TGT_ENC   = "#9b59b6"   # purple (target encoder)
C_LOSS      = "#e74c3c"   # red (loss)
C_BG        = "#1a1a2e"   # dark navy bg
C_TEXT      = "#ecf0f1"   # light text
C_ACCENT    = "#f1c40f"   # yellow accent
C_VIT_REP   = "#d35400"   # dark orange (ViT representation)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def make_textbox(text, color=C_TEXT, font_size=28, **kwargs):
    return Text(text, color=color, font_size=font_size, **kwargs)


def make_mathbox(tex, color=C_TEXT, font_size=28, **kwargs):
    return MathTex(tex, color=color, font_size=font_size, **kwargs)


def make_block(
    label, width=2.5, height=1.0, fill_color=C_CTXT_ENC, fill_opacity=0.85,
    stroke_color=WHITE, stroke_width=1.5, label_color=BLACK, font_size=20,
):
    rect = RoundedRectangle(
        width=width, height=height, corner_radius=0.12,
        fill_color=fill_color, fill_opacity=fill_opacity,
        stroke_color=stroke_color, stroke_width=stroke_width,
    )
    txt = Text(label, color=label_color, font_size=font_size, weight=BOLD)
    return VGroup(rect, txt)


def arrow_between(start, end, color=WHITE, buff=0.15):
    return Arrow(
        start=start, end=end, color=color, buff=buff, stroke_width=3,
        max_tip_length_to_length_ratio=0.12,
    )


def tensor_label(tensor_shape, color=C_INPUT, font_size=18):
    return MathTex(tensor_shape, color=color, font_size=font_size)


# ---------------------------------------------------------------------------
# Scene 1 — Title & Problem Context
# ---------------------------------------------------------------------------
class TitleScene(Scene):
    config = {
        "background_color": C_BG,
    }

    def construct(self):
        # ── Title ──
        title = Text(
            "V-JEPA", font_size=72, color=C_ACCENT, weight=BOLD
        )
        subtitle = Text(
            "Video Joint Embedding Predictive Architecture",
            font_size=28, color=C_TEXT,
        )
        subtitle.next_to(title, DOWN, buff=0.3)

        conf_line = Text(
            "ICLR 2024  ·  Meta AI / FAIR", font_size=22, color=GREY,
        )
        conf_line.next_to(subtitle, DOWN, buff=0.4)

        logo_group = VGroup(title, subtitle, conf_line)
        logo_group.center()
        self.play(Write(title), Write(subtitle), Write(conf_line), run_time=2)
        self.wait(0.5)
        self.play(logo_group.animate.scale(0.6).to_edge(UP + LEFT, buff=0.3))

        # ── Problem → Solution contrast ──
        self.next_section("Problem vs Approach")

        left_box = Rectangle(
            width=4.5, height=3.0, color=C_PROBLEM, fill_opacity=0.15,
            stroke_width=2,
        )
        left_box.shift(LEFT * 2.8 + DOWN * 0.2)
        left_title = Text("Pixel Reconstruction", font_size=22, color=C_PROBLEM, weight=BOLD)
        left_title.next_to(left_box, UP, buff=0.2)
        left_desc = Text(
            "Example: VideoMAE\n\nReconstruct masked\npixels → decoder overhead,\nsemantically weak loss",
            font_size=16, color=C_TEXT, line_spacing=0.6,
        )
        left_desc.move_to(left_box)

        right_box = Rectangle(
            width=4.5, height=3.0, color=C_SOLUTION, fill_opacity=0.15,
            stroke_width=2,
        )
        right_box.shift(RIGHT * 2.8 + DOWN * 0.2)
        right_title = Text("Latent Space Prediction", font_size=22, color=C_SOLUTION, weight=BOLD)
        right_title.next_to(right_box, UP, buff=0.2)
        right_desc = Text(
            "V-JEPA\n\nPredict masked latents\n→ no pixel decoder,\nsemantically rich loss",
            font_size=16, color=C_TEXT, line_spacing=0.6,
        )
        right_desc.move_to(right_box)

        vs_label = Text("vs", font_size=30, color=C_ACCENT, slant=ITALIC)
        vs_label.move_to(ORIGIN + DOWN * 0.2)

        self.play(
            FadeIn(left_box), Write(left_title), Write(left_desc),
            FadeIn(right_box), Write(right_title), Write(right_desc),
            Write(vs_label),
            run_time=2,
        )
        self.wait(1)

        # ── Tagline ──
        tagline = Text(
            "Self-supervised visual representation learning from video\n"
            "without pixel-level reconstruction",
            font_size=20, color=C_TEXT, line_spacing=0.7,
        )
        tagline.to_edge(DOWN, buff=0.6)
        self.play(Write(tagline))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------------------------------------
# Scene 2 — 3D Patch Tokenization
# ---------------------------------------------------------------------------
class PatchTokenizeScene(Scene):
    config = {"background_color": C_BG}

    def construct(self):
        # ── Section label ──
        section = make_textbox("Input Processing & 3D Patch Tokenization", color=C_ACCENT, font_size=30)
        section.to_edge(UP, buff=0.4)
        underline = Underline(section, color=C_ACCENT)
        self.play(Write(section), GrowFromCenter(underline))

        # ── Input video clip representation ──
        input_label = make_textbox("Input Video Clip", font_size=18, color=C_TEXT)
        input_label.shift(UP * 1.2 + LEFT * 4.2)

        # Layered video representation (3D cube)
        cube_width, cube_height, cube_depth = 2.4, 2.4, 2.0
        video_cube = Cube(
            side_length=2.0, fill_color=C_INPUT, fill_opacity=0.3,
            stroke_color=C_INPUT, stroke_width=2,
        )
        video_cube.move_to(LEFT * 4.0 + UP * 0.2)

        # Shape annotation
        shape_tex = MathTex(
            r"16 \times 224 \times 224 \times 3",
            font_size=22, color=C_INPUT,
        )
        shape_tex.next_to(video_cube, DOWN, buff=0.3)

        # Axes labels
        t_label = Text("T=16 frames", font_size=14, color=GREY)
        t_label.next_to(video_cube, UP, buff=0.15)
        h_label = Text("H=224", font_size=14, color=GREY)
        h_label.next_to(video_cube, RIGHT, buff=0.15)
        w_label = Text("W=224", font_size=14, color=GREY)
        w_label.next_to(video_cube, DOWN + LEFT, buff=0.15).shift(RIGHT * 0.3)

        input_group = VGroup(video_cube, shape_tex, t_label, h_label, w_label, input_label)
        self.play(FadeIn(video_cube), Write(shape_tex), Write(t_label), Write(h_label), Write(w_label), Write(input_label))
        self.wait(0.5)

        # ── 3D Convolution arrow ──
        conv_arrow = Arrow(
            start=video_cube.get_right() + RIGHT * 0.3,
            end=video_cube.get_right() + RIGHT * 1.8,
            color=C_ACCENT, buff=0.1, stroke_width=3,
        )
        conv_label = make_textbox(
            "3D Conv\n2×16×16\nstride 2×16×16",
            font_size=14, color=C_ACCENT,
        )
        conv_label.next_to(conv_arrow, UP, buff=0.2)
        self.play(GrowArrow(conv_arrow), Write(conv_label))
        self.wait(0.5)

        # ── Token sequence representation ──
        # Show grid of small squares representing 1568 tokens
        token_grid = VGroup()
        rows, cols = 8, 14   # 8 temporal × 14 spatial = 112 per time step, but we'll show a 8×14 grid
        square_size = 0.25
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=square_size, color=C_TOKEN, fill_opacity=0.7, stroke_width=0)
                sq.move_to(RIGHT * 2.5 + UP * 1.8 + RIGHT * c * (square_size + 0.04) + DOWN * r * (square_size + 0.04))
                token_grid.add(sq)
        token_grid.center().shift(RIGHT * 2.5 + UP * 0.2)

        token_label_lines = [
            MathTex(r"8 \times 14 \times 14 = 1568 \text{ patches}", font_size=20, color=C_TOKEN),
            MathTex(r"[1568 \times d] \text{ tokens } (d=1024 \text{ for ViT-L})", font_size=18, color=C_TOKEN),
        ]
        token_label_1 = token_label_lines[0].next_to(token_grid, DOWN, buff=0.4)
        token_label_2 = token_label_lines[1].next_to(token_label_1, DOWN, buff=0.15)

        # Pos embed note
        pos_note = make_textbox("+ 3D Sincos Positional Embedding", font_size=14, color=GREY)
        pos_note.next_to(token_label_2, DOWN, buff=0.25)

        self.play(
            LaggedStart(*[Create(sq) for sq in token_grid], lag_ratio=0.001, run_time=1.5),
            Write(token_label_1), Write(token_label_2), Write(pos_note),
        )
        self.wait(0.5)

        # ── Code reference ──
        code_ref = make_textbox(
            "src/models/utils/patch_embed.py → PatchEmbed3D\n"
            "src/models/vision_transformer.py → VisionTransformer",
            font_size=12, color=GREY,
        )
        code_ref.to_edge(DOWN, buff=0.3)
        self.play(Write(code_ref))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------------------------------------
# Scene 3 — 3D Multi-Block Masking
# ---------------------------------------------------------------------------
class MultiBlockMaskScene(Scene):
    config = {"background_color": C_BG}

    def construct(self):
        # ── Section label ──
        section = make_textbox("3D Multi-Block Masking Strategy", color=C_ACCENT, font_size=30)
        section.to_edge(UP, buff=0.4)
        underline = Underline(section, color=C_ACCENT)
        self.play(Write(section), GrowFromCenter(underline))

        # ── Left: Short-range mask ──
        short_title = make_textbox("Short-Range Mask", font_size=22, color=C_TEXT, weight=BOLD)
        short_title.shift(LEFT * 3.0 + UP * 2.2)
        short_params = make_textbox(
            "8 blocks  ·  scale 0.15\naspect ratio [0.75, 1.5]",
            font_size=16, color=GREY,
        )
        short_params.next_to(short_title, DOWN, buff=0.15)

        # Draw a 3D grid (8×14×14) simplified to 2D projections for clarity
        short_grid = self.draw_mask_grid(
            rows=8, cols=14, n_blocks=8, scale=0.15,
            top_left=LEFT * 3.8 + UP * 0.8,
            cell_size=0.28,
        )

        short_pct = make_textbox("~85% masked → ~15% kept", font_size=14, color=C_PROBLEM)
        short_pct.next_to(short_grid, DOWN, buff=0.25)

        self.play(Write(short_title), Write(short_params), FadeIn(short_grid), Write(short_pct))
        self.wait(0.5)

        # ── Right: Long-range mask ──
        long_title = make_textbox("Long-Range Mask", font_size=22, color=C_TEXT, weight=BOLD)
        long_title.shift(RIGHT * 3.0 + UP * 2.2)
        long_params = make_textbox(
            "2 blocks  ·  scale 0.7\naspect ratio [0.75, 1.5]",
            font_size=16, color=GREY,
        )
        long_params.next_to(long_title, DOWN, buff=0.15)

        long_grid = self.draw_mask_grid(
            rows=8, cols=14, n_blocks=2, scale=0.7,
            top_left=RIGHT * 2.2 + UP * 0.8,  # offset right by half-width
            cell_size=0.28,
        )

        long_pct = make_textbox("~70% masked → ~30% kept", font_size=14, color=C_SOLUTION)
        long_pct.next_to(long_grid, DOWN, buff=0.25)

        self.play(Write(long_title), Write(long_params), FadeIn(long_grid), Write(long_pct))
        self.wait(0.5)

        # ── Bottom: Multi-mask explanation ──
        bottom_text = make_textbox(
            "Both masks are applied simultaneously —\n"
            "the encoder and predictor receive multiple mask views per clip",
            font_size=18, color=C_TEXT, line_spacing=0.5,
        )
        bottom_text.to_edge(DOWN, buff=0.8)
        self.play(Write(bottom_text))

        # ── Output separation visualization ──
        sep_label = make_textbox(
            "Output:  encoder_masks (kept token indices)  +  predictor_masks (masked token indices)",
            font_size=16, color=C_ACCENT,
        )
        sep_label.next_to(bottom_text, DOWN, buff=0.2)
        self.play(Write(sep_label))

        code_ref = make_textbox(
            "src/masks/multiblock3d.py → MaskCollator / _MaskGenerator",
            font_size=12, color=GREY,
        )
        code_ref.to_edge(DOWN, buff=0.15)
        self.play(Write(code_ref))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def draw_mask_grid(self, rows, cols, n_blocks=8, scale=0.15, top_left=None, cell_size=0.28):
        """Draw a simplified 2D representation of the 3D mask grid.
        Randomly mark cells as kept (green) or masked (red).
        """
        import random
        random.seed(42)
        grid = VGroup()
        if top_left is None:
            top_left = ORIGIN
        for r in range(rows):
            for c in range(cols):
                sq = Square(
                    side_length=cell_size, stroke_width=0.5,
                    color=C_MASK_BG, fill_opacity=0.3,
                )
                sq.move_to(
                    top_left +
                    RIGHT * c * (cell_size + 0.02) +
                    DOWN * r * (cell_size + 0.02)
                )
                # Simple heuristic: keep ratio ≈ scale
                if random.random() < scale:
                    sq.set_fill(C_MASK_KEEP, opacity=0.7)   # kept context
                else:
                    sq.set_fill(C_MASK_PRED, opacity=0.7)   # masked / target
                grid.add(sq)
        return grid


# ---------------------------------------------------------------------------
# Scene 4 — Core Architecture & Latent Prediction Pipeline
# ---------------------------------------------------------------------------
class ArchitectureScene(Scene):
    config = {"background_color": C_BG}

    def construct(self):
        # ── Title ──
        title = make_textbox("V-JEPA Architecture: Latent Prediction Pipeline", font_size=30, color=C_ACCENT)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.3)

        # ── Layout constants ──
        left_x = -5.5
        mid_x = -1.5
        right_x = 3.5

        # ── Step 0: Input clip ──
        input_box = make_block(
            "Input\n16×224×224×3", width=2.2, height=1.2,
            fill_color=C_INPUT, font_size=14, label_color=WHITE,
        )
        input_box.move_to(UP * 2.5 + left_x * 0.5)

        input_label = VGroup(
            make_textbox("Video Clip", font_size=14, color=C_TEXT),
            tensor_label(r"[B,3,T,H,W]", font_size=12, color=GREY),
        ).arrange(DOWN, buff=0.05).next_to(input_box, DOWN, buff=0.15)
        self.play(FadeIn(input_box), Write(input_label))

        # ── 3D Patch Embed ──
        embed_box = make_block(
            "3D Conv\nPatchEmbed\n2×16×16", width=2.0, height=1.3,
            fill_color=C_TOKEN, font_size=13, label_color=WHITE,
        )
        embed_box.move_to(UP * 3.0 + mid_x * 1.5)

        arrow_input_embed = Arrow(
            input_box.get_right() + RIGHT * 0.2,
            embed_box.get_left() + LEFT * 0.2,
            color=WHITE, buff=0.1,
        )
        self.play(GrowArrow(arrow_input_embed))
        self.play(FadeIn(embed_box))
        self.wait(0.2)

        # Token output annotation
        token_ann = make_textbox(
            "1568 tokens × d", font_size=13, color=C_TOKEN,
        )
        token_ann.next_to(embed_box, DOWN, buff=0.15)
        self.play(Write(token_ann))

        # ── Masking step ──
        mask_box = make_block(
            "3D Multi-Block\nMasking\nshort + long", width=2.2, height=1.3,
            fill_color=C_MASK_BG, font_size=13, label_color=WHITE,
        )
        mask_box.next_to(embed_box, DOWN, buff=0.6)

        arrow_embed_mask = Arrow(
            embed_box.get_bottom() + DOWN * 0.15,
            mask_box.get_top() + UP * 0.15,
            color=WHITE, buff=0.1,
        )
        self.play(GrowArrow(arrow_embed_mask), FadeIn(mask_box))

        # Split annotation
        split_label = VGroup(
            make_textbox("Unmasked (N × d)", font_size=12, color=C_MASK_KEEP),
            make_textbox("Masked (M × d)", font_size=12, color=C_MASK_PRED),
        ).arrange(RIGHT, buff=0.4).next_to(mask_box, DOWN, buff=0.2)
        self.play(Write(split_label))

        # ── LEFT BRANCH: Context Encoder ──
        ctxt_enc_box = make_block(
            "Context\nEncoder\nE_θ  (ViT-L/16)", width=2.3, height=1.5,
            fill_color=C_CTXT_ENC, font_size=14, label_color=BLACK,
        )
        ctxt_enc_box.move_to(DOWN * 0.6 + left_x)

        arrow_mask_ctxt = Arrow(
            mask_box.get_left() + LEFT * 0.5,
            ctxt_enc_box.get_top() + UP * 0.3,
            color=C_MASK_KEEP, buff=0.1,
        )
        ctxt_in = make_textbox("unmasked tokens", font_size=11, color=C_MASK_KEEP)
        ctxt_in.next_to(arrow_mask_ctxt, LEFT, buff=0.1).rotate(PI / 2)
        self.play(GrowArrow(arrow_mask_ctxt), FadeIn(ctxt_enc_box), Write(ctxt_in))
        self.wait(0.2)

        # Context encoder output
        ctxt_out = make_mathbox(r"z \in \mathbb{R}^{N \times d}", font_size=16, color=C_CTXT_ENC)
        ctxt_out.next_to(ctxt_enc_box, DOWN, buff=0.15)
        self.play(Write(ctxt_out))

        # ── RIGHT BRANCH: Target Encoder ──
        tgt_enc_box = make_block(
            "Target\nEncoder\nE_θ̄  (EMA stop-grad)", width=2.3, height=1.5,
            fill_color=C_TGT_ENC, font_size=14, label_color=WHITE,
        )
        tgt_enc_box.shift(DOWN * 0.6 + right_x)

        arrow_mask_tgt = Arrow(
            mask_box.get_right() + RIGHT * 0.5,
            tgt_enc_box.get_top() + UP * 0.3,
            color=C_MASK_PRED, buff=0.1,
        )
        tgt_in = make_textbox("full unmasked\nclip (L × d)", font_size=11, color=C_TEXT)
        tgt_in.next_to(arrow_mask_tgt, RIGHT, buff=0.1).rotate(PI / 2)
        self.play(GrowArrow(arrow_mask_tgt), FadeIn(tgt_enc_box), Write(tgt_in))
        self.wait(0.2)

        # EMA annotation
        ema_note = make_textbox(
            "EMA: θ̄ ← m·θ̄ + (1−m)·θ\nm ∈ [0.998, 1.0]",
            font_size=11, color=GREY,
        )
        ema_note.next_to(tgt_enc_box, DOWN, buff=0.15)

        # Target encoder output: mask selects the target positions
        tgt_out = make_mathbox(r"s_M \in \mathbb{R}^{M \times d}", font_size=16, color=C_TGT_ENC)
        tgt_out.next_to(ema_note, DOWN, buff=0.15)
        self.play(Write(ema_note), Write(tgt_out))

        # ── CENTER BRANCH: Predictor ──
        pred_box = make_block(
            "Predictor\nP_φ  (narrow ViT)\n+ mask tokens", width=2.5, height=1.6,
            fill_color=C_PREDICTOR, font_size=14, label_color=BLACK,
        )
        pred_box.move_to(DOWN * 2.5 + mid_x * 0.7)

        arrow_ctxt_pred = Arrow(
            ctxt_out.get_bottom() + DOWN * 0.2,
            pred_box.get_top() + UP * 0.2 + LEFT * 1.0,
            color=C_CTXT_ENC, buff=0.1,
        )
        self.play(GrowArrow(arrow_ctxt_pred))
        self.play(FadeIn(pred_box))
        self.wait(0.2)

        # Predictor details
        pred_detail = VGroup(
            make_textbox("Concatenate z + mask tokens → predict", font_size=11, color=C_TEXT),
            make_textbox("Predictor dim: 384  (narrow, depth: 12)", font_size=11, color=GREY),
            make_textbox("Output: ŝ_M  (predicted latents)", font_size=11, color=C_PREDICTOR),
        ).arrange(DOWN, buff=0.1).next_to(pred_box, DOWN, buff=0.15)
        self.play(Write(pred_detail))

        # ── LOSS ──
        loss_box = make_block(
            "L₁ Loss\nΣ|ŝ_M − s_M|", width=2.0, height=1.2,
            fill_color=C_LOSS, font_size=16, label_color=WHITE,
        )
        loss_box.move_to(DOWN * 3.6)

        # Arrow from predictor output to loss
        arrow_pred_loss = Arrow(
            pred_box.get_bottom() + DOWN * 0.8 + RIGHT * 0.5,
            loss_box.get_top() + UP * 0.2 + LEFT * 0.5,
            color=C_PREDICTOR, buff=0.1,
        )
        # Arrow from target output to loss
        arrow_tgt_loss = Arrow(
            tgt_out.get_bottom() + DOWN * 0.5,
            loss_box.get_top() + UP * 0.2 + RIGHT * 0.5,
            color=C_TGT_ENC, buff=0.1,
        )
        self.play(GrowArrow(arrow_pred_loss), GrowArrow(arrow_tgt_loss), FadeIn(loss_box))
        self.wait(0.5)

        # ── STOP-GRADIENT annotation ──
        sg_box = DashedVMobject(
            Rectangle(width=3.6, height=3.0, color=C_TGT_ENC, stroke_width=2),
            num_dashes=20,
        )
        sg_box.move_to(tgt_enc_box)
        sg_note = make_textbox("Stop-Gradient", font_size=13, color=C_TGT_ENC, weight=BOLD)
        sg_note.next_to(sg_box, RIGHT, buff=0.3)
        self.play(Create(sg_box), Write(sg_note))
        self.wait(1)

        # ── Final annotations ──
        code_ref = make_textbox(
            "Predictor:  src/models/predictor.py → VisionTransformerPredictor\n"
            "Encoder:    src/models/vision_transformer.py → VisionTransformer\n"
            "Training:   app/vjepa/train.py → forward_context / forward_target / loss_fn",
            font_size=10, color=GREY, line_spacing=0.4,
        )
        code_ref.to_edge(DOWN, buff=0.15)
        self.play(Write(code_ref))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------------------------------------
# Scene 5 — Key Highlights & Results
# ---------------------------------------------------------------------------
class HighlightsScene(Scene):
    config = {"background_color": C_BG}

    def construct(self):
        # ── Title ──
        title = make_textbox("Key Results & Highlights", font_size=34, color=C_ACCENT, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # ── Column 1: Architecture highlights ──
        arch_title = make_textbox("Architecture", font_size=22, color=C_TEXT, weight=BOLD)
        arch_title.shift(LEFT * 3.2 + UP * 1.0)

        arch_items = VGroup(*[
            make_textbox(f"  {item}", font_size=16, color=C_TEXT)
            for item in [
                "No pixel decoder — latent prediction",
                "3D Multi-Block masking (~90% ratio)",
                "EMA target encoder (momentum update)",
                "Narrow ViT predictor (d=384, depth=12)",
                "Learnable mask tokens + diffusion noise",
                "Simple L₁ loss in latent space",
            ]
        ])
        arch_items.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        arch_items.next_to(arch_title, DOWN, buff=0.3, aligned_edge=LEFT)
        arch_items.shift(LEFT * 3.2)

        for item in arch_items:
            item.align_to(arch_items[0], LEFT)

        self.play(Write(arch_title))
        self.play(LaggedStart(*[Write(item) for item in arch_items], lag_ratio=0.2, run_time=2))
        self.wait(0.3)

        # ── Column 2: Downstream results ──
        res_title = make_textbox("Downstream Performance (ViT-L/16)", font_size=22, color=C_TEXT, weight=BOLD)
        res_title.shift(RIGHT * 2.5 + UP * 1.0)

        res_items = VGroup(
            self.result_row("Kinetics-400", "80.8%", "video classification", C_MASK_KEEP),
            self.result_row("Something-Something v2", "69.5%", "motion understanding", C_MASK_KEEP),
            self.result_row("ImageNet-1K", "74.8%", "image classification", C_MASK_KEEP),
            self.result_row("Places205", "60.3%", "scene recognition", C_TEXT),
            self.result_row("iNaturalist 2021", "67.8%", "fine-grained", C_TEXT),
        )
        res_items.arrange(DOWN, buff=0.25)
        res_items.next_to(res_title, DOWN, buff=0.3)
        res_items.shift(RIGHT * 2.5)

        self.play(Write(res_title))
        self.play(LaggedStart(*[FadeIn(item) for item in res_items], lag_ratio=0.15, run_time=2))
        self.wait(0.5)

        # ── Bottom: Key insight ──
        insight = Text(
            '"All evaluations use frozen backbone + lightweight Attentive Probe — no fine-tuning."',
            font_size=18, color=C_ACCENT, line_spacing=0.6,
        )
        insight.to_edge(DOWN, buff=0.7)
        insight_box = SurroundingRectangle(insight, color=C_ACCENT, buff=0.25, corner_radius=0.1)
        self.play(Write(insight), Create(insight_box))
        self.wait(2)

        # ── Credits ──
        credits = make_textbox(
            "Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas — ICLR 2024",
            font_size=14, color=GREY,
        )
        credits.to_edge(DOWN, buff=0.2)
        self.play(Write(credits))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def result_row(self, name, score, description, color):
        row = VGroup(
            Text(f"{name}:  ", font_size=16, color=color, weight=BOLD),
            Text(score, font_size=16, color=color),
            Text(f"  ({description})", font_size=13, color=GREY),
        )
        row.arrange(RIGHT, buff=0.05)
        return row


# ---------------------------------------------------------------------------
# All-in-one combined scene
# ---------------------------------------------------------------------------
class VJEPAAllScenes(Scene):
    """Single scene that composites all sub-scenes using Manim sections."""

    config = {"background_color": C_BG}

    def construct(self):
        # ────── SECTION 1: Title & Problem Context ──────
        self.next_section("Title & Problem")

        title = Text("V-JEPA", font_size=72, color=C_ACCENT, weight=BOLD)
        subtitle = Text(
            "Video Joint Embedding Predictive Architecture",
            font_size=28, color=C_TEXT,
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        conf_line = Text(
            "ICLR 2024  ·  Meta AI / FAIR", font_size=22, color=GREY,
        )
        conf_line.next_to(subtitle, DOWN, buff=0.4)
        logo_group = VGroup(title, subtitle, conf_line)
        logo_group.center()
        self.play(Write(title), Write(subtitle), Write(conf_line), run_time=2)
        self.wait(0.3)
        self.play(logo_group.animate.scale(0.6).to_edge(UP + LEFT, buff=0.3))

        left_box = Rectangle(width=4.5, height=3.0, color=C_PROBLEM, fill_opacity=0.15, stroke_width=2)
        left_box.shift(LEFT * 2.8 + DOWN * 0.2)
        left_title = Text("Pixel Reconstruction", font_size=22, color=C_PROBLEM, weight=BOLD)
        left_title.next_to(left_box, UP, buff=0.2)
        left_desc = Text(
            "Example: VideoMAE\n\nReconstruct masked\npixels → decoder overhead,\nsemantically weak loss",
            font_size=16, color=C_TEXT, line_spacing=0.6,
        )
        left_desc.move_to(left_box)
        right_box = Rectangle(width=4.5, height=3.0, color=C_SOLUTION, fill_opacity=0.15, stroke_width=2)
        right_box.shift(RIGHT * 2.8 + DOWN * 0.2)
        right_title = Text("Latent Space Prediction", font_size=22, color=C_SOLUTION, weight=BOLD)
        right_title.next_to(right_box, UP, buff=0.2)
        right_desc = Text(
            "V-JEPA\n\nPredict masked latents\n→ no pixel decoder,\nsemantically rich loss",
            font_size=16, color=C_TEXT, line_spacing=0.6,
        )
        right_desc.move_to(right_box)
        vs_label = Text("vs", font_size=30, color=C_ACCENT, slant=ITALIC)
        vs_label.move_to(ORIGIN + DOWN * 0.2)
        tagline = Text(
            "Self-supervised visual representation learning from video\nwithout pixel-level reconstruction",
            font_size=20, color=C_TEXT, line_spacing=0.7,
        )
        tagline.to_edge(DOWN, buff=0.6)

        self.play(
            FadeIn(left_box), Write(left_title), Write(left_desc),
            FadeIn(right_box), Write(right_title), Write(right_desc),
            Write(vs_label), run_time=2,
        )
        self.play(Write(tagline))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ────── SECTION 2: 3D Patch Tokenization ──────
        self.next_section("3D Patch Tokenization")
        self._draw_patch_tokenization()
        # ────── SECTION 3: 3D Multi-Block Masking ──────
        self.next_section("Multi-Block Masking")
        self._draw_multiblock_masking()
        # ────── SECTION 4: Architecture ──────
        self.next_section("Architecture Pipeline")
        self._draw_architecture()
        # ────── SECTION 5: Results ──────
        self.next_section("Results")
        self._draw_highlights()

    # -- helper methods ------------------------------------------------------

    def _draw_patch_tokenization(self):
        section = make_textbox("Input Processing & 3D Patch Tokenization", color=C_ACCENT, font_size=30)
        section.to_edge(UP, buff=0.4)
        underline = Underline(section, color=C_ACCENT)
        self.play(Write(section), GrowFromCenter(underline))

        input_label = make_textbox("Input Video Clip", font_size=18, color=C_TEXT)
        input_label.shift(UP * 1.2 + LEFT * 4.2)
        video_cube = Cube(side_length=2.0, fill_color=C_INPUT, fill_opacity=0.3, stroke_color=C_INPUT, stroke_width=2)
        video_cube.move_to(LEFT * 4.0 + UP * 0.2)
        shape_tex = MathTex(r"16 \times 224 \times 224 \times 3", font_size=22, color=C_INPUT)
        shape_tex.next_to(video_cube, DOWN, buff=0.3)
        t_label = Text("T=16 frames", font_size=14, color=GREY)
        t_label.next_to(video_cube, UP, buff=0.15)
        h_label = Text("H=224", font_size=14, color=GREY)
        h_label.next_to(video_cube, RIGHT, buff=0.15)
        w_label = Text("W=224", font_size=14, color=GREY)
        w_label.next_to(video_cube, DOWN + LEFT, buff=0.15).shift(RIGHT * 0.3)

        self.play(FadeIn(video_cube), Write(shape_tex), Write(t_label), Write(h_label), Write(w_label), Write(input_label))
        self.wait(0.3)

        conv_arrow = Arrow(video_cube.get_right() + RIGHT * 0.3, video_cube.get_right() + RIGHT * 1.8, color=C_ACCENT, buff=0.1, stroke_width=3)
        conv_label = make_textbox("3D Conv\n2×16×16\nstride 2×16×16", font_size=14, color=C_ACCENT)
        conv_label.next_to(conv_arrow, UP, buff=0.2)
        self.play(GrowArrow(conv_arrow), Write(conv_label))

        token_grid = VGroup()
        rows, cols = 8, 14
        square_size = 0.25
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=square_size, color=C_TOKEN, fill_opacity=0.7, stroke_width=0)
                sq.move_to(RIGHT * 2.5 + UP * 1.8 + RIGHT * c * (square_size + 0.04) + DOWN * r * (square_size + 0.04))
                token_grid.add(sq)
        token_grid.center().shift(RIGHT * 2.5 + UP * 0.2)
        token_label_1 = MathTex(r"8 \times 14 \times 14 = 1568 \text{ patches}", font_size=20, color=C_TOKEN)
        token_label_2 = MathTex(r"[1568 \times d] \text{ tokens } (d=1024 \text{ for ViT-L})", font_size=18, color=C_TOKEN)
        token_label_1.next_to(token_grid, DOWN, buff=0.4)
        token_label_2.next_to(token_label_1, DOWN, buff=0.15)
        pos_note = make_textbox("+ 3D Sincos Positional Embedding", font_size=14, color=GREY)
        pos_note.next_to(token_label_2, DOWN, buff=0.25)

        self.play(
            LaggedStart(*[Create(sq) for sq in token_grid], lag_ratio=0.001, run_time=1.2),
            Write(token_label_1), Write(token_label_2), Write(pos_note),
        )
        code_ref = make_textbox(
            "src/models/utils/patch_embed.py → PatchEmbed3D",
            font_size=12, color=GREY,
        )
        code_ref.to_edge(DOWN, buff=0.3)
        self.play(Write(code_ref))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def _draw_multiblock_masking(self):
        section = make_textbox("3D Multi-Block Masking Strategy", color=C_ACCENT, font_size=30)
        section.to_edge(UP, buff=0.4)
        underline = Underline(section, color=C_ACCENT)
        self.play(Write(section), GrowFromCenter(underline))

        short_title = make_textbox("Short-Range Mask", font_size=22, color=C_TEXT, weight=BOLD)
        short_title.shift(LEFT * 3.0 + UP * 2.2)
        short_params = make_textbox("8 blocks  ·  scale 0.15\naspect ratio [0.75, 1.5]", font_size=16, color=GREY)
        short_params.next_to(short_title, DOWN, buff=0.15)
        short_grid = self._make_mask_grid(rows=8, cols=14, n_blocks=8, scale=0.15, top_left=LEFT * 3.8 + UP * 0.8, cell_size=0.28)
        short_pct = make_textbox("~85% masked → ~15% kept", font_size=14, color=C_PROBLEM)
        short_pct.next_to(short_grid, DOWN, buff=0.25)

        self.play(Write(short_title), Write(short_params), FadeIn(short_grid), Write(short_pct))
        self.wait(0.3)

        long_title = make_textbox("Long-Range Mask", font_size=22, color=C_TEXT, weight=BOLD)
        long_title.shift(RIGHT * 3.0 + UP * 2.2)
        long_params = make_textbox("2 blocks  ·  scale 0.7\naspect ratio [0.75, 1.5]", font_size=16, color=GREY)
        long_params.next_to(long_title, DOWN, buff=0.15)
        long_grid = self._make_mask_grid(rows=8, cols=14, n_blocks=2, scale=0.7, top_left=RIGHT * 2.2 + UP * 0.8, cell_size=0.28)
        long_pct = make_textbox("~70% masked → ~30% kept", font_size=14, color=C_SOLUTION)
        long_pct.next_to(long_grid, DOWN, buff=0.25)

        self.play(Write(long_title), Write(long_params), FadeIn(long_grid), Write(long_pct))
        self.wait(0.3)

        bottom_text = make_textbox(
            "Both masks are applied simultaneously — encoder & predictor receive multiple mask views per clip",
            font_size=18, color=C_TEXT, line_spacing=0.5,
        )
        bottom_text.to_edge(DOWN, buff=0.8)
        self.play(Write(bottom_text))
        sep_label = make_textbox(
            "Output:  encoder_masks (kept idxs)  +  predictor_masks (masked idxs)",
            font_size=16, color=C_ACCENT,
        )
        sep_label.next_to(bottom_text, DOWN, buff=0.2)
        self.play(Write(sep_label))
        code_ref = make_textbox(
            "src/masks/multiblock3d.py → MaskCollator / _MaskGenerator",
            font_size=12, color=GREY,
        )
        code_ref.to_edge(DOWN, buff=0.15)
        self.play(Write(code_ref))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def _draw_architecture(self):
        title = make_textbox("V-JEPA Architecture: Latent Prediction Pipeline", font_size=30, color=C_ACCENT)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        left_x, mid_x, right_x = -5.5, -1.5, 3.5

        input_box = make_block("Input\n16×224×224×3", width=2.2, height=1.2, fill_color=C_INPUT, font_size=14, label_color=WHITE)
        input_box.move_to(UP * 2.5 + left_x * 0.5)
        input_label = VGroup(
            make_textbox("Video Clip", font_size=14, color=C_TEXT),
            tensor_label(r"[B,3,T,H,W]", font_size=12, color=GREY),
        ).arrange(DOWN, buff=0.05).next_to(input_box, DOWN, buff=0.15)
        self.play(FadeIn(input_box), Write(input_label))

        embed_box = make_block("3D Conv\nPatchEmbed\n2×16×16", width=2.0, height=1.3, fill_color=C_TOKEN, font_size=13, label_color=WHITE)
        embed_box.move_to(UP * 3.0 + mid_x * 1.5)
        arrow_input_embed = Arrow(input_box.get_right() + RIGHT * 0.2, embed_box.get_left() + LEFT * 0.2, color=WHITE, buff=0.1, stroke_width=2)
        self.play(GrowArrow(arrow_input_embed), FadeIn(embed_box))
        token_ann = make_textbox("1568 tokens × d", font_size=13, color=C_TOKEN)
        token_ann.next_to(embed_box, DOWN, buff=0.15)
        self.play(Write(token_ann))

        mask_box = make_block("3D Multi-Block\nMasking", width=2.2, height=1.3, fill_color=C_MASK_BG, font_size=13, label_color=WHITE)
        mask_box.next_to(embed_box, DOWN, buff=0.6)
        arrow_embed_mask = Arrow(embed_box.get_bottom() + DOWN * 0.15, mask_box.get_top() + UP * 0.15, color=WHITE, buff=0.1, stroke_width=2)
        self.play(GrowArrow(arrow_embed_mask), FadeIn(mask_box))
        split_label = VGroup(
            make_textbox("Unmasked (N × d)", font_size=12, color=C_MASK_KEEP),
            make_textbox("Masked (M × d)", font_size=12, color=C_MASK_PRED),
        ).arrange(RIGHT, buff=0.4).next_to(mask_box, DOWN, buff=0.2)
        self.play(Write(split_label))

        ctxt_enc_box = make_block("Context\nEncoder\nE_θ  (ViT-L/16)", width=2.3, height=1.5, fill_color=C_CTXT_ENC, font_size=14, label_color=BLACK)
        ctxt_enc_box.move_to(DOWN * 0.6 + left_x)
        arrow_mask_ctxt = Arrow(
            mask_box.get_left() + LEFT * 0.5, ctxt_enc_box.get_top() + UP * 0.3,
            color=C_MASK_KEEP, buff=0.1, stroke_width=2,
        )
        self.play(GrowArrow(arrow_mask_ctxt), FadeIn(ctxt_enc_box))
        self.wait(0.2)
        ctxt_out = make_mathbox(r"z \in \mathbb{R}^{N \times d}", font_size=16, color=C_CTXT_ENC)
        ctxt_out.next_to(ctxt_enc_box, DOWN, buff=0.15)
        self.play(Write(ctxt_out))

        tgt_enc_box = make_block("Target\nEncoder\nE_θ̄  (EMA)", width=2.3, height=1.5, fill_color=C_TGT_ENC, font_size=14, label_color=WHITE)
        tgt_enc_box.shift(DOWN * 0.6 + right_x)
        arrow_mask_tgt = Arrow(
            mask_box.get_right() + RIGHT * 0.5, tgt_enc_box.get_top() + UP * 0.3,
            color=C_TEXT, buff=0.1, stroke_width=2,
        )
        self.play(GrowArrow(arrow_mask_tgt), FadeIn(tgt_enc_box))
        self.wait(0.2)
        ema_note = make_textbox("θ̄ ← m·θ̄ + (1−m)·θ\nm ∈ [0.998, 1.0]", font_size=11, color=GREY)
        ema_note.next_to(tgt_enc_box, DOWN, buff=0.15)
        tgt_out = make_mathbox(r"s_M \in \mathbb{R}^{M \times d}", font_size=16, color=C_TGT_ENC)
        tgt_out.next_to(ema_note, DOWN, buff=0.15)
        self.play(Write(ema_note), Write(tgt_out))

        pred_box = make_block("Predictor\nP_φ  (narrow ViT)\n+ mask tokens", width=2.5, height=1.6, fill_color=C_PREDICTOR, font_size=14, label_color=BLACK)
        pred_box.move_to(DOWN * 2.5 + mid_x * 0.7)
        arrow_ctxt_pred = Arrow(ctxt_out.get_bottom() + DOWN * 0.2, pred_box.get_top() + UP * 0.2 + LEFT * 1.0, color=C_CTXT_ENC, buff=0.1, stroke_width=2)
        self.play(GrowArrow(arrow_ctxt_pred), FadeIn(pred_box))
        self.wait(0.2)
        pred_detail = VGroup(
            make_textbox("[z + mask_tokens] → predict → ŝ_M", font_size=12, color=C_TEXT),
            make_textbox("Pred dim: 384, depth: 12  (narrow ViT)", font_size=11, color=GREY),
        ).arrange(DOWN, buff=0.1).next_to(pred_box, DOWN, buff=0.15)
        self.play(Write(pred_detail))

        loss_box = make_block("L₁ Loss\nΣ|ŝ_M − s_M|", width=2.0, height=1.2, fill_color=C_LOSS, font_size=16, label_color=WHITE)
        loss_box.move_to(DOWN * 3.6)
        arrow_pred_loss = Arrow(
            pred_box.get_bottom() + DOWN * 0.8 + RIGHT * 0.5,
            loss_box.get_top() + UP * 0.2 + LEFT * 0.5, color=C_PREDICTOR, buff=0.1, stroke_width=2,
        )
        arrow_tgt_loss = Arrow(
            tgt_out.get_bottom() + DOWN * 0.5,
            loss_box.get_top() + UP * 0.2 + RIGHT * 0.5, color=C_TGT_ENC, buff=0.1, stroke_width=2,
        )
        self.play(GrowArrow(arrow_pred_loss), GrowArrow(arrow_tgt_loss), FadeIn(loss_box))
        self.wait(0.3)

        sg_box = DashedVMobject(Rectangle(width=3.6, height=3.0, color=C_TGT_ENC, stroke_width=2), num_dashes=20)
        sg_box.move_to(tgt_enc_box)
        sg_note = make_textbox("Stop-Gradient", font_size=13, color=C_TGT_ENC, weight=BOLD)
        sg_note.next_to(sg_box, RIGHT, buff=0.3)
        self.play(Create(sg_box), Write(sg_note))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def _draw_highlights(self):
        title = make_textbox("Key Results & Highlights", font_size=34, color=C_ACCENT, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        arch_title = make_textbox("Architecture", font_size=22, color=C_TEXT, weight=BOLD)
        arch_title.shift(LEFT * 3.2 + UP * 1.0)
        arch_items = VGroup(*[
            make_textbox(f"  {item}", font_size=16, color=C_TEXT)
            for item in [
                "No pixel decoder — latent prediction",
                "3D Multi-Block masking (~90% ratio)",
                "EMA target encoder (momentum update)",
                "Narrow ViT predictor (d=384, depth=12)",
                "Learnable mask tokens + diffusion noise",
                "Simple L₁ loss in latent space",
            ]
        ])
        arch_items.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        arch_items.next_to(arch_title, DOWN, buff=0.3, aligned_edge=LEFT)
        arch_items.shift(LEFT * 3.2)
        for item in arch_items:
            item.align_to(arch_items[0], LEFT)
        self.play(Write(arch_title))
        self.play(LaggedStart(*[Write(item) for item in arch_items], lag_ratio=0.2, run_time=2))
        self.wait(0.3)

        res_title = make_textbox("Downstream Performance (ViT-L/16)", font_size=22, color=C_TEXT, weight=BOLD)
        res_title.shift(RIGHT * 2.5 + UP * 1.0)
        res_items = VGroup(
            self._result_row("Kinetics-400", "80.8%", "video class.", C_MASK_KEEP),
            self._result_row("Something-Something v2", "69.5%", "motion", C_MASK_KEEP),
            self._result_row("ImageNet-1K", "74.8%", "image class.", C_MASK_KEEP),
            self._result_row("Places205", "60.3%", "scenes", C_TEXT),
            self._result_row("iNaturalist 2021", "67.8%", "fine-grained", C_TEXT),
        )
        res_items.arrange(DOWN, buff=0.25)
        res_items.next_to(res_title, DOWN, buff=0.3)
        res_items.shift(RIGHT * 2.5)
        self.play(Write(res_title))
        self.play(LaggedStart(*[FadeIn(item) for item in res_items], lag_ratio=0.15, run_time=2))
        self.wait(0.5)

        insight = Text(
            'All evaluations use frozen backbone + lightweight Attentive Probe — no fine-tuning.',
            font_size=18, color=C_ACCENT, line_spacing=0.6,
        )
        insight.to_edge(DOWN, buff=0.7)
        insight_box = SurroundingRectangle(insight, color=C_ACCENT, buff=0.25, corner_radius=0.1)
        self.play(Write(insight), Create(insight_box))
        credits = make_textbox(
            "Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas — ICLR 2024",
            font_size=14, color=GREY,
        )
        credits.to_edge(DOWN, buff=0.2)
        self.play(Write(credits))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    @staticmethod
    def _result_row(name, score, description, color):
        return VGroup(
            Text(f"{name}:  ", font_size=16, color=color, weight=BOLD),
            Text(score, font_size=16, color=color),
            Text(f"  ({description})", font_size=13, color=GREY),
        ).arrange(RIGHT, buff=0.05)

    @staticmethod
    def _make_mask_grid(rows, cols, n_blocks, scale, top_left, cell_size):
        import random
        random.seed(42)
        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell_size, stroke_width=0.5, color=C_MASK_BG, fill_opacity=0.3)
                sq.move_to(top_left + RIGHT * c * (cell_size + 0.02) + DOWN * r * (cell_size + 0.02))
                if random.random() < scale:
                    sq.set_fill(C_MASK_KEEP, opacity=0.7)
                else:
                    sq.set_fill(C_MASK_PRED, opacity=0.7)
                grid.add(sq)
        return grid
