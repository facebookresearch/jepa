"""
V-JEPA Chapters 5, 6, 7 — Enhanced Manim Animation
Style: 3Blue1Brown — dark background, vector math, smooth animations

Prerequisites:
    1. Run tts_ch567_full.py first to generate audio files
    2. conda activate jepa

Render (480p preview):
    cd c:\\jepa\\jepa\\visualizations
    manim -ql vjepa_ch567_full.py VJEPACh567Scene

Static layout preview:
    manim -ql -s vjepa_ch567_full.py VJEPACh567Scene

Single chapter test:
    manim -ql vjepa_ch567_full.py Ch5Scene
    manim -ql vjepa_ch567_full.py Ch6Scene
    manim -ql vjepa_ch567_full.py Ch7Scene
"""

import os
os.environ["PATH"] += os.pathsep + r"C:\\Users\\tony\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64"

from manim import *
import numpy as np

# ── Global font override: use Arial for all Text() in this file ────────────
# Manim defaults to CMU Serif on Windows; override to get a clean sans-serif.
_OrigText = Text
class Text(_OrigText):  # noqa: F811
    """Drop-in replacement for manim.Text that defaults to Arial."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("font", "Arial")
        super().__init__(*args, **kwargs)

# ── Try importing mutagen; fall back to fixed durations if unavailable ────
try:
    from mutagen.mp3 import MP3
    def _get_dur(path: str) -> float:
        try:
            return MP3(path).info.length
        except Exception:
            return 55.0
except ImportError:
    def _get_dur(path: str) -> float:
        return 55.0

# ═════════════════════════════════════════════════════════════════════════
# COLOR PALETTE  (3B1B dark style)
# ═════════════════════════════════════════════════════════════════════════
BG        = "#0e1117"   # near-black background
C_BLUE    = "#3B82F6"   # Context Encoder
C_ORANGE  = "#F97316"   # Predictor
C_SILVER  = "#9CA3AF"   # Target Encoder / dim
C_RED     = "#EF4444"   # Stop-gradient / danger
C_YELLOW  = "#F59E0B"   # Gradient arrows / accent
C_GREEN   = "#10B981"   # EMA / success / probe
C_TEXT    = "#F1F5F9"   # Main white text
C_DIM     = "#6B7280"   # Secondary / dim text
C_PURPLE  = "#8B5CF6"   # Pooled representation
C_TEAL    = "#06B6D4"   # Misc accent / scene


# ═════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════
def sync_wait(scene, t0: float, dur: float, label: str = "") -> None:
    """Wait until the audio track finishes."""
    elapsed  = scene.renderer.time - t0
    remaining = dur - elapsed
    if remaining > 0.05:
        scene.wait(remaining)
    else:
        print(f"  [WARNING] [{label}] animation ({elapsed:.1f}s) > audio ({dur:.1f}s)")


def safe_sound(scene, path: str, is_master: bool = False) -> float:
    """Add sound and return duration; safe-fail if file missing."""
    if os.path.exists(path):
        scene.add_sound(path)   # always add — works in both single-chapter and master renders
        return _get_dur(path)
    print(f"  [WARNING] Audio not found: {path} - using silent placeholder")
    return 55.0


def wait_to(scene, t0: float, target_sec: float, label: str = "") -> None:
    """Wait until target_sec seconds have elapsed since t0 (narration sync anchor)."""
    elapsed = scene.renderer.time - t0
    remaining = target_sec - elapsed
    if remaining > 0.02:
        scene.wait(remaining)
    elif remaining < -0.3:
        print(f"  [WARNING] [{label}] animation overran by {-remaining:.1f}s at {target_sec:.1f}s mark")


def chapter_header(num: int, title: str, color=C_YELLOW) -> VGroup:
    """Returns a positioned chapter title VGroup."""
    ttl = Text(title, font_size=27, color=color, weight=BOLD)
    grp = VGroup(ttl)
    grp.to_edge(UP, buff=0.35)
    return grp


def rounded_box(w: float, h: float, color: str,
                fill_op: float = 0.22, stroke_w: float = 2.5) -> RoundedRectangle:
    return RoundedRectangle(
        width=w, height=h, corner_radius=0.12,
        fill_color=color, fill_opacity=fill_op,
        stroke_color=color, stroke_width=stroke_w,
    ).set_z_index(2)


def labeled_box(label_lines: list, box_w: float, box_h: float,
                color: str, font_sizes: list = None) -> VGroup:
    """Rounded rect with 1 or 2 lines of text centered inside."""
    if font_sizes is None:
        font_sizes = [18] * len(label_lines)
    rect = rounded_box(box_w, box_h, color)
    texts = VGroup(*[
        Text(line, font_size=sz, color=color, weight=BOLD if i == 0 else NORMAL)
        for i, (line, sz) in enumerate(zip(label_lines, font_sizes))
    ]).arrange(DOWN, buff=0.1).set_z_index(3)
    # center text in rect (both at ORIGIN before positioning)
    texts.move_to(rect)
    return VGroup(rect, texts)


def make_arrow(start, end, color=C_TEXT, sw=2.2) -> Arrow:
    return Arrow(
        start, end, color=color, buff=0.08,
        stroke_width=sw, max_tip_length_to_length_ratio=0.13,
    ).set_z_index(1)


# ── Subtitle helper ─────────────────────────────────────────────────────
def make_sub(text: str, font_size: int = 14) -> VGroup:
    """Semi-transparent subtitle bar pinned to the bottom of the frame."""
    return VGroup()


# ═════════════════════════════════════════════════════════════════════════
# CHAPTER 5 SCENE
# ═════════════════════════════════════════════════════════════════════════
class Ch5Scene(Scene):
    def construct(self):
        self.camera.background_color = BG
        self._ch5_1_loss_stopgrad()
        self._ch5_2_collapse_ema()

    # ─────────────────────────────────────────────────────────────────────
    # 5.1  Loss Function + Stop-Gradient
    # ─────────────────────────────────────────────────────────────────────
    def _ch5_1_loss_stopgrad(self, is_master: bool = False):
        AUDIO = "media/ch5_1_new.mp3"
        dur = safe_sound(self, AUDIO, is_master=is_master)
        t0  = self.renderer.time

        # ── Phase 1: Section title ───────────────────────────────────────
        hdr  = chapter_header(5, "Mathematical Heart — Loss & Stop-Gradient")
        div  = Line(LEFT * 6.0, RIGHT * 6.0, color=C_YELLOW, stroke_width=1.2)
        div.next_to(hdr, DOWN, buff=0.1)
        self.play(FadeIn(hdr, shift=DOWN * 0.25), GrowFromCenter(div), run_time=0.9)

        # ── t=0:04  Phase 2: Loss formula ────────────────────────────────
        # Narration: "We compute the smooth L-1 distance..."
        wait_to(self, t0, 4.0, "ch5_1 formula")
        _sub = make_sub("Loss L: L1 distance between predictor output and target encoder output.")
        self.add(_sub)
        f_lbl = Text("V-JEPA Prediction Loss  (L1 in latent space):",
                     font_size=18, color=C_DIM)
        f_lbl.next_to(div, DOWN, buff=0.38)

        loss_eq = MathTex(
            r"\mathcal{L} = \dfrac{1}{M}\sum_{k=1}^{M}"
            r"\bigl\|\,\hat{s}_k - s_k\,\bigr\|_1",
            font_size=44,
        ).set_color(C_YELLOW).next_to(f_lbl, DOWN, buff=0.3)

        hat_leg = VGroup(
            MathTex(r"\hat{s}_k", font_size=22, color=C_ORANGE),
            Text("= Predictor output  (predicted latent)", font_size=15, color=C_ORANGE),
        ).arrange(RIGHT, buff=0.12)
        s_leg = VGroup(
            MathTex(r"s_k", font_size=22, color=C_SILVER),
            Text("= Target Encoder output  (stop-gradient target)", font_size=15, color=C_SILVER),
        ).arrange(RIGHT, buff=0.12)
        legend = VGroup(hat_leg, s_leg).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        legend.next_to(loss_eq, DOWN, buff=0.32)

        self.play(FadeIn(f_lbl), Write(loss_eq), run_time=1.6)
        self.play(FadeIn(legend, lag_ratio=0.4), run_time=0.9)
        self.wait(0.3)

        # ── t=0:14  Slide formula to corner, show architecture ───────────
        # Narration: "This is a fundamentally different objective..."
        self.remove(_sub)
        _sub = make_sub("V-JEPA predicts abstract latent representations — not raw pixels like other methods.")
        self.add(_sub)
        wait_to(self, t0, 14.0, "ch5_1 arch")
        self.play(
            FadeOut(hdr), FadeOut(div), FadeOut(f_lbl), FadeOut(legend),
            loss_eq.animate.scale(0.68).to_corner(UL, buff=0.45).shift(DOWN * 0.4),
            run_time=0.9,
        )

        # ── Phase 3: Architecture diagram ────────────────────────────────
        # Positions
        CTX_POS  = LEFT  * 3.8 + UP   * 0.85
        PRED_POS = RIGHT * 1.4 + UP   * 0.85
        TGT_POS  = LEFT  * 3.8 + DOWN * 1.0
        LOSS_POS = RIGHT * 4.5 + DOWN * 0.08

        ctx_grp  = labeled_box(["Context Encoder"], 2.9, 0.92, C_BLUE)
        pred_grp = labeled_box(["Predictor"],        2.4, 0.92, C_ORANGE)
        tgt_grp  = labeled_box(["Target Encoder"],   2.9, 0.92, C_SILVER,
                                font_sizes=[18])

        ctx_grp.move_to(CTX_POS)
        pred_grp.move_to(PRED_POS)
        tgt_grp.move_to(TGT_POS)
        # Re-center text inside each box after positioning
        ctx_grp[1].move_to(ctx_grp[0])
        pred_grp[1].move_to(pred_grp[0])
        tgt_grp[1].move_to(tgt_grp[0])

        loss_circ = Circle(
            radius=0.44, fill_color=C_RED, fill_opacity=0.28,
            stroke_color=C_RED, stroke_width=2.5,
        ).move_to(LOSS_POS).set_z_index(2)
        loss_circ_lbl = MathTex(r"\mathcal{L}", font_size=34, color=C_RED)
        loss_circ_lbl.move_to(loss_circ).set_z_index(3)
        loss_node = VGroup(loss_circ, loss_circ_lbl)

        arch = VGroup(ctx_grp, pred_grp, tgt_grp, loss_node)
        self.play(FadeIn(arch, lag_ratio=0.22), run_time=1.2)

        # ── t=0:22  Forward-pass arrows ──────────────────────────────────
        # Narration: "Now watch the gradient flow during backpropagation."
        self.remove(_sub)
        _sub = make_sub("Watch how gradients flow backward through the architecture during training.")
        self.add(_sub)
        wait_to(self, t0, 22.0, "ch5_1 fwd")
        ctx_rect  = ctx_grp[0]
        pred_rect = pred_grp[0]
        tgt_rect  = tgt_grp[0]

        fwd_cp = make_arrow(ctx_rect.get_right(),  pred_rect.get_left())
        fwd_pl = make_arrow(pred_rect.get_right(), loss_circ.get_left())
        fwd_tl = make_arrow(tgt_rect.get_right(),  loss_circ.get_bottom() + DOWN * 0.05)
        fwds   = VGroup(fwd_cp, fwd_pl, fwd_tl)
        self.play(
            AnimationGroup(*[GrowArrow(a) for a in fwds], lag_ratio=0.3),
            run_time=1.2,
        )

        # ── t=0:26  Phase 4: Gradient backpropagation ────────────────────
        # Narration: "Gradients propagate backward from the loss..."
        self.remove(_sub)
        _sub = make_sub("Gradients flow: Loss → Predictor → Context Encoder (weights are updated).")
        self.add(_sub)
        wait_to(self, t0, 26.0, "ch5_1 grad")
        grad_note = VGroup(
            MathTex(r"\nabla", font_size=20, color=C_YELLOW),
            Text("  gradient flows backward", font_size=16, color=C_YELLOW),
        ).arrange(RIGHT, buff=0.08)
        grad_note.to_corner(UR, buff=0.5).set_z_index(4)
        self.play(FadeIn(grad_note, shift=LEFT * 0.2), run_time=0.5)

        g_lp = Arrow(
            loss_circ.get_left(), pred_rect.get_right() + RIGHT * 0.05,
            color=C_YELLOW, buff=0.1, stroke_width=3.2,
            max_tip_length_to_length_ratio=0.14,
        ).set_z_index(4)
        g_pc = Arrow(
            pred_rect.get_left(), ctx_rect.get_right() + RIGHT * 0.05,
            color=C_YELLOW, buff=0.1, stroke_width=3.2,
            max_tip_length_to_length_ratio=0.14,
        ).set_z_index(4)
        self.play(GrowArrow(g_lp), run_time=0.7)
        self.play(GrowArrow(g_pc), run_time=0.7)

        # ── t=0:33  Stop-gradient ─────────────────────────────────────────
        # Narration: "when the gradient signal reaches the Target Encoder boundary"
        self.remove(_sub)
        _sub = make_sub("Stop-gradient: the gradient is completely blocked at the Target Encoder boundary.")
        self.add(_sub)
        wait_to(self, t0, 33.0, "ch5_1 stopgrad")

        # Gradient attempt toward target encoder — BLOCKED
        loss_bot = loss_circ.get_bottom() + DOWN * 0.05
        tgt_right = tgt_rect.get_right()
        midpt = (loss_bot + tgt_right) * 0.5

        g_partial = Arrow(
            loss_bot, midpt,
            color=C_YELLOW, buff=0.05, stroke_width=2.6,
            max_tip_length_to_length_ratio=0.2,
        ).set_z_index(4)

        stop_wall = Line(
            midpt + UP * 0.52, midpt + DOWN * 0.52,
            color=C_RED, stroke_width=11,
        ).set_z_index(6)
        x_mark = MathTex(r"\times", font_size=48, color=C_RED).move_to(midpt).set_z_index(7)
        stop_lbl = Text("STOP\nGRADIENT", font_size=13, color=C_RED, weight=BOLD)
        stop_lbl.next_to(stop_wall, UP, buff=0.12).set_z_index(8)

        self.play(GrowArrow(g_partial), run_time=0.5)
        self.play(
            FadeIn(stop_wall, scale=1.9),
            FadeIn(x_mark, scale=1.9),
            FadeIn(stop_lbl),
            run_time=0.65, rate_func=smooth,
        )
        # Pulse effect
        self.play(stop_wall.animate.set_stroke(width=16), run_time=0.22)
        self.play(stop_wall.animate.set_stroke(width=11), run_time=0.22)

        # ── t=0:44  EMA annotation ────────────────────────────────────────
        # Narration: "Without it, the model could trivially set both encoders..."
        self.remove(_sub)
        _sub = make_sub("Without stop-gradient, both encoders collapse to constants — loss = 0, learning = 0.")
        self.add(_sub)
        wait_to(self, t0, 44.0, "ch5_1 ema")
        ema_note = Text("Updated via EMA only  →", font_size=14, color=C_SILVER)
        ema_note.next_to(tgt_rect, DOWN, buff=0.2).set_z_index(3)
        self.play(FadeIn(ema_note, shift=UP * 0.15), run_time=0.6)

        all_51 = VGroup(
            loss_eq, arch, fwds,
            g_lp, g_pc, g_partial,
            stop_wall, x_mark, stop_lbl, grad_note, ema_note,
        )
        self.remove(_sub)
        sync_wait(self, t0, dur, "ch5_1")
        self.play(FadeOut(all_51), run_time=0.8)

    # ─────────────────────────────────────────────────────────────────────
    # 5.2  Representation Collapse + EMA
    # ─────────────────────────────────────────────────────────────────────
    def _ch5_2_collapse_ema(self, is_master: bool = False):
        AUDIO = "media/ch5_2_new.mp3"
        dur = safe_sound(self, AUDIO, is_master=is_master)
        t0  = self.renderer.time

        # ── Phase 1: Collapse title ──────────────────────────────────────
        col_title = Text("Representation Collapse", font_size=38, color=C_RED, weight=BOLD)
        col_title.to_edge(UP, buff=0.4)
        col_sub = Text(
            "The silent enemy of self-supervised learning",
            font_size=20, color=C_DIM,
        ).next_to(col_title, DOWN, buff=0.16)
        self.play(FadeIn(col_title, shift=DOWN * 0.3), FadeIn(col_sub), run_time=0.9)

        # ── t=0:07  Phase 2: Diverse video dots ──────────────────────────
        # Narration: "Imagine every possible video input — a running athlete..."
        wait_to(self, t0, 7.0, "ch5_2 dots")
        _sub = make_sub("All possible video inputs — athletes, cooking, nature — as distinct points in space.")
        self.add(_sub)
        np.random.seed(7)
        palette = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_TEAL, "#EC4899", "#F87171", "#A78BFA"]
        n = 52
        xs = np.random.uniform(-5.4, 5.4, n)
        ys = np.random.uniform(-2.4, 1.8, n)
        dots = VGroup(*[
            Dot(radius=0.09, color=palette[i % len(palette)], fill_opacity=0.9)
            .move_to([xs[i], ys[i], 0])
            for i in range(n)
        ]).set_z_index(2)

        type_labels = VGroup(
            Text("running", font_size=13, color=C_DIM).move_to([-3.5, 1.5, 0]),
            Text("cooking", font_size=13, color=C_DIM).move_to([1.2, -1.8, 0]),
            Text("nature",  font_size=13, color=C_DIM).move_to([4.0, 1.2, 0]),
            Text("sports",  font_size=13, color=C_DIM).move_to([-1.5, -2.0, 0]),
        ).set_z_index(1)

        self.play(
            FadeIn(dots, lag_ratio=0.008, run_time=1.4),
            FadeIn(type_labels, lag_ratio=0.3, run_time=1.4),
        )
        self.wait(0.5)

        # ── Phase 3: Collapse ────────────────────────────────────────────
        collapse_dot = Dot(radius=0.48, color=C_RED, fill_opacity=1.0)
        collapse_dot.move_to(ORIGIN).set_z_index(5)
        trivial_lbl = Text(
            "Trivial Solution  —  Loss = 0,   Learning = 0",
            font_size=22, color=C_RED, weight=BOLD,
        ).next_to(collapse_dot, DOWN, buff=0.35).set_z_index(5)

        # ── t=0:16  Phase 3: Collapse ──────────────────────────────────────
        # Narration: "The loss drops to zero, yet the model has learned absolutely nothing."
        self.remove(_sub)
        _sub = make_sub("Collapse: loss = 0, but the model learned nothing — all inputs map to one point.")
        self.add(_sub)
        wait_to(self, t0, 16.0, "ch5_2 collapse")
        self.play(
            ReplacementTransform(dots, collapse_dot),
            FadeOut(type_labels),
            run_time=1.8, rate_func=smooth,
        )
        self.play(FadeIn(trivial_lbl, shift=UP * 0.2), run_time=0.7)

        # ── t=0:21  Phase 4: EMA Solution ────────────────────────────────
        # Narration: "V-JEPA escapes this trap through two mathematically interlocked mechanisms."
        self.remove(_sub)
        _sub = make_sub("V-JEPA uses two interlocked mechanisms to prevent collapse.")
        self.add(_sub)
        wait_to(self, t0, 21.0, "ch5_2 ema-hdr")
        self.play(FadeOut(VGroup(col_title, col_sub, collapse_dot, trivial_lbl)), run_time=0.7)

        # ── Phase 4: EMA Solution header ─────────────────────────────────
        ema_title = Text(
            "Solution: Exponential Moving Average (EMA)",
            font_size=30, color=C_GREEN, weight=BOLD,
        ).to_edge(UP, buff=0.4)
        self.play(FadeIn(ema_title, shift=DOWN * 0.2), run_time=0.7)

        ema_eq = MathTex(
            r"\bar{\theta}_t \;\leftarrow\; m \cdot \bar{\theta}_{t-1} + (1-m)\cdot\theta_t",
            font_size=44,
        ).set_color(C_GREEN).next_to(ema_title, DOWN, buff=0.48)
        self.play(Write(ema_eq), run_time=1.5)

        # Term legend
        leg_tbar = VGroup(
            MathTex(r"\bar{\theta}", font_size=22, color=C_SILVER),
            Text("Target Encoder weights  (slow, stable)", font_size=15, color=C_SILVER),
        ).arrange(RIGHT, buff=0.14)
        leg_t = VGroup(
            MathTex(r"\theta", font_size=22, color=C_BLUE),
            Text("Context Encoder weights  (actively learning)", font_size=15, color=C_BLUE),
        ).arrange(RIGHT, buff=0.14)
        leg_m = VGroup(
            MathTex(r"m", font_size=22, color=C_YELLOW),
            Text("Momentum:  0.998  →  1.0  (linear schedule)", font_size=15, color=C_YELLOW),
        ).arrange(RIGHT, buff=0.14)
        legend = VGroup(leg_tbar, leg_t, leg_m).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        legend.next_to(ema_eq, DOWN, buff=0.38)
        # ── t=0:35  Legend ────────────────────────────────────────────────
        # Narration: "Instead of training the Target Encoder directly..."
        self.remove(_sub)
        _sub = make_sub("EMA: Target Encoder = slow moving average of Context Encoder weights.")
        self.add(_sub)
        wait_to(self, t0, 35.0, "ch5_2 legend")
        self.play(FadeIn(legend, lag_ratio=0.35), run_time=1.0)

        # ── t=0:43  Phase 5: Momentum schedule ───────────────────────────
        # Narration: "The momentum m starts at 0.998 and gradually increases..."
        self.remove(_sub)
        _sub = make_sub("Momentum m: 0.998 to 1.0 over training, making target increasingly stable.")
        self.add(_sub)
        wait_to(self, t0, 43.0, "ch5_2 mom")
        self.play(FadeOut(legend), run_time=0.5)

        ax_line = Line(LEFT * 3.2, RIGHT * 3.2, color=C_DIM, stroke_width=1.3)
        ax_line.next_to(ema_eq, DOWN, buff=0.52)
        track = Line(ax_line.get_left(), ax_line.get_right(), color=C_YELLOW, stroke_width=3.5)
        track.put_start_and_end_on(ax_line.get_left(), ax_line.get_left())   # start as point

        d_start = Dot(radius=0.09, color=C_YELLOW).move_to(ax_line.get_left())
        d_end   = Dot(radius=0.09, color=C_YELLOW).move_to(ax_line.get_right())
        lbl_s   = MathTex(r"m{=}0.998", font_size=17, color=C_YELLOW).next_to(d_start, UP, buff=0.1)
        lbl_e   = MathTex(r"m{\to}1.0",  font_size=17, color=C_YELLOW).next_to(d_end,   UP, buff=0.1)
        x_lbl   = Text("Training iterations  →", font_size=13, color=C_DIM)
        x_lbl.next_to(ax_line, DOWN, buff=0.18)

        self.play(
            Create(ax_line), FadeIn(d_start), FadeIn(lbl_s), FadeIn(x_lbl),
            run_time=0.8,
        )
        self.play(
            track.animate.put_start_and_end_on(ax_line.get_left(), ax_line.get_right()),
            run_time=1.4, rate_func=smooth,
        )
        self.play(FadeIn(d_end), FadeIn(lbl_e), run_time=0.5)
        self.wait(0.4)

        mom_group = VGroup(ax_line, track, d_start, d_end, lbl_s, lbl_e, x_lbl)
        self.play(FadeOut(mom_group), run_time=0.5)

        # ── t=0:54  Phase 6: MAD theorem ─────────────────────────────────
        # Narration: "The mathematics here is elegant: when the Predictor achieves optimality..."
        self.remove(_sub)
        _sub = make_sub("Optimal Predictor under L1 -> gradient = MAD, collapse is mathematically impossible.")
        self.add(_sub)
        wait_to(self, t0, 54.0, "ch5_2 mad")
        mad_ctx = VGroup(
            Text("When Predictor is optimal under ", font_size=20, color=C_TEXT),
            MathTex(r"L_1", font_size=24, color=C_TEXT),
            Text(" loss:", font_size=20, color=C_TEXT),
        ).arrange(RIGHT, buff=0.05).next_to(ema_eq, DOWN, buff=0.42)
        mad_eq1 = MathTex(
            r"p^\star(z_N) = \mathrm{median}\bigl(X \mid z_N(\theta)\bigr)",
            font_size=30,
        ).set_color(C_ORANGE).next_to(mad_ctx, DOWN, buff=0.28)
        down_arr = MathTex(r"\Downarrow", font_size=34, color=C_YELLOW)
        down_arr.next_to(mad_eq1, DOWN, buff=0.2)
        mad_eq2 = MathTex(
            r"\nabla_\theta\,\mathcal{L} \;=\; "
            r"\nabla_\theta \sum_{l=1}^{d} \mathrm{MAD}\!\bigl(X_l \mid z_N(\theta)\bigr)",
            font_size=26,
        ).set_color(C_GREEN).next_to(down_arr, DOWN, buff=0.2)
        mad_note = Text(
            "Encoder is forced to capture rich diverse representations  →  collapse is impossible",
            font_size=15, color=C_DIM,
        ).next_to(mad_eq2, DOWN, buff=0.2)

        self.play(FadeIn(mad_ctx), run_time=0.5)
        self.play(Write(mad_eq1), run_time=1.1)
        self.play(FadeIn(down_arr), run_time=0.3)
        self.play(Write(mad_eq2), run_time=1.4)
        self.play(FadeIn(mad_note), run_time=0.5)

        all_52 = VGroup(ema_title, ema_eq, mad_ctx, mad_eq1, down_arr, mad_eq2, mad_note)
        self.remove(_sub)
        sync_wait(self, t0, dur, "ch5_2")
        self.play(FadeOut(all_52), run_time=0.8)


# ═════════════════════════════════════════════════════════════════════════
# CHAPTER 6 SCENE
# ═════════════════════════════════════════════════════════════════════════
class Ch6Scene(Scene):
    def construct(self):
        self.camera.background_color = BG
        self._ch6_1_benchmarks()
        self._ch6_2_efficiency()

    # ─────────────────────────────────────────────────────────────────────
    # 6.1  Benchmark Comparison
    # ─────────────────────────────────────────────────────────────────────
    def _ch6_1_benchmarks(self, is_master: bool = False):
        AUDIO = "media/ch6_1_new.mp3"
        dur = safe_sound(self, AUDIO, is_master=is_master)
        t0  = self.renderer.time

        hdr = chapter_header(6, "V-JEPA vs. The Competition")
        self.play(FadeIn(hdr, shift=DOWN * 0.2), run_time=0.8)

        # ── t=0:04  Charts appear ─────────────────────────────────────────
        # Narration: "First, Something-Something-v2 — the most demanding test..."
        wait_to(self, t0, 4.0, "ch6_1 charts")
        _sub = make_sub("SSv2: the most demanding benchmark for physical motion understanding.")
        self.add(_sub)

        # ── SSv2 chart (left) ────────────────────────────────────────────
        ssv2_vals  = [71.2, 61.2, 60.3, 50.0, 39.0]
        ssv2_names = ["V-JEPA", "VideoMAE", "InternVideo", "DINOv2", "OpenCLIP"]
        C_BAR_OTHER = "#9CA3AF"   # brighter than C_DIM for readable bar labels
        ssv2_cols  = [C_ORANGE, C_BAR_OTHER, C_BAR_OTHER, C_BAR_OTHER, C_BAR_OTHER]
        ssv2_chart = BarChart(
            values=ssv2_vals, bar_names=ssv2_names,
            y_range=[0, 80, 20], x_length=5.5, y_length=2.8,
            bar_colors=ssv2_cols,
            bar_width=0.55,
            x_axis_config={"font_size": 14, "color": C_TEXT},
            y_axis_config={"color": C_TEXT},
        ).set_z_index(1)
        ssv2_head = Text(
            "SSv2  —  Physical Motion Understanding",
            font_size=16, color=C_TEXT, weight=BOLD,
        ).next_to(ssv2_chart, UP, buff=0.2)
        ssv2_grp = VGroup(ssv2_head, ssv2_chart)

        # ── K400 chart (right) ───────────────────────────────────────────
        k400_vals  = [82.1, 77.9, 73.7]
        k400_names = ["V-JEPA", "VideoMAE", "InternVideo"]
        k400_cols  = [C_ORANGE, C_BAR_OTHER, C_BAR_OTHER]
        k400_chart = BarChart(
            values=k400_vals, bar_names=k400_names,
            y_range=[0, 90, 30], x_length=3.2, y_length=2.8,
            bar_colors=k400_cols,
            bar_width=0.55,
            x_axis_config={"font_size": 14, "color": C_TEXT},
            y_axis_config={"color": C_TEXT},
        ).set_z_index(1)
        k400_head = Text(
            "K400  —  Appearance & Context",
            font_size=16, color=C_TEXT, weight=BOLD,
        ).next_to(k400_chart, UP, buff=0.2)
        k400_grp = VGroup(k400_head, k400_chart)

        charts = VGroup(ssv2_grp, k400_grp).arrange(RIGHT, buff=0.9)
        charts.next_to(hdr, DOWN, buff=0.44)

        # Animate charts with stagger
        self.play(
            AnimationGroup(
                AnimationGroup(FadeIn(ssv2_head), Create(ssv2_chart), lag_ratio=0.15),
                AnimationGroup(FadeIn(k400_head), Create(k400_chart), lag_ratio=0.15),
                lag_ratio=0.4,
            ),
            run_time=2.0,
        )

        # ── t=0:18  Value labels + highlight ─────────────────────────────
        # Narration: "V-JEPA achieves 71.2 percent accuracy..."
        self.remove(_sub)
        _sub = make_sub("V-JEPA: 71.2% on SSv2 and 82.1% on K400 — state-of-the-art with a frozen backbone.")
        self.add(_sub)
        wait_to(self, t0, 18.0, "ch6_1 labels")
        # Value labels
        ssv2_lbls = ssv2_chart.get_bar_labels(
            font_size=14,
            label_constructor=lambda v: Text(f"{v}%", font_size=13),
        )
        k400_lbls = k400_chart.get_bar_labels(
            font_size=14,
            label_constructor=lambda v: Text(f"{v}%", font_size=13),
        )
        self.play(
            FadeIn(ssv2_lbls, lag_ratio=0.1),
            FadeIn(k400_lbls, lag_ratio=0.1),
            run_time=0.7,
        )

        # ── t=0:25  Highlight V-JEPA bars ────────────────────────────────
        # Narration: "This is a remarkable 10.0 percentage point improvement..."
        self.remove(_sub)
        _sub = make_sub("+10.0 pts over VideoMAE and +32.2 pts over OpenCLIP on SSv2 — a remarkable margin.")
        self.add(_sub)
        wait_to(self, t0, 25.0, "ch6_1 highlight")
        self.play(Indicate(ssv2_chart.bars[0], color=C_YELLOW, scale_factor=1.05), run_time=0.5)
        self.play(Indicate(k400_chart.bars[0], color=C_YELLOW, scale_factor=1.05), run_time=0.5)

        # ── t=0:40  Delta annotations ──────────────────────────────────
        # Narration: "The gap on Something-Something-v2 is the most telling result..."
        self.remove(_sub)
        _sub = make_sub("Largest gaps on motion-heavy tasks — exactly where latent prediction excels.")
        self.add(_sub)
        wait_to(self, t0, 40.0, "ch6_1 delta")
        delta = Text(
            "+10.0 pts vs VideoMAE   •   +32.2 pts vs OpenCLIP   (SSv2, frozen eval)",
            font_size=16, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.42)
        source = Text(
            "Source: V-JEPA, Meta AI Research, 2024  —  Frozen backbone evaluation protocol",
            font_size=11, color=C_DIM,
        ).next_to(delta, DOWN, buff=0.1)
        self.play(FadeIn(delta, shift=UP * 0.2), FadeIn(source), run_time=0.8)

        all_61 = VGroup(hdr, charts, ssv2_lbls, k400_lbls, delta, source)
        self.remove(_sub)
        sync_wait(self, t0, dur, "ch6_1")
        self.play(FadeOut(all_61), run_time=0.8)

    # ─────────────────────────────────────────────────────────────────────
    # 6.2  Controlled Comparison + Sample Efficiency
    # ─────────────────────────────────────────────────────────────────────
    def _ch6_2_efficiency(self, is_master: bool = False):
        AUDIO = "media/ch6_2_new.mp3"
        dur = safe_sound(self, AUDIO, is_master=is_master)
        t0  = self.renderer.time

        hdr = chapter_header(6, "Controlled Comparison & Sample Efficiency")
        self.play(FadeIn(hdr, shift=DOWN * 0.2), run_time=0.7)

        # ── t=0:05  Phase A: Controlled experiment boxes ──────────────────
        # Narration: "We fix the architecture to identical ViT-L 16 models..."
        wait_to(self, t0, 5.0, "ch6_2 ctrl")
        _sub = make_sub("Fair comparison: same ViT-L/16, same K400 data — only pretraining objective differs.")
        self.add(_sub)
        ctrl_lbl = Text(
            "Fair comparison:  same ViT-L/16 architecture,  trained only on K400",
            font_size=18, color=C_TEXT,
        ).next_to(hdr, DOWN, buff=0.42)

        vj_grp = labeled_box(["V-JEPA"], 2.5, 1.0, C_ORANGE)
        vs_txt  = Text("vs", font_size=24, color=C_DIM)
        vm_grp  = labeled_box(["VideoMAE"], 2.5, 1.0, C_SILVER)

        boxes_row = VGroup(vj_grp, vs_txt, vm_grp).arrange(RIGHT, buff=0.75)
        boxes_row.next_to(ctrl_lbl, DOWN, buff=0.42)
        # re-center text
        vj_grp[1].move_to(vj_grp[0])
        vm_grp[1].move_to(vm_grp[0])

        deltas = VGroup(
            Text("+0.7 pts  on  Kinetics-400",       font_size=19, color=C_GREEN),
            Text("+0.5 pts  on  SSv2",               font_size=19, color=C_GREEN),
            Text("+3.4 pts  on  AVA action detection", font_size=19, color=C_GREEN),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        deltas.next_to(boxes_row, DOWN, buff=0.42)

        self.play(FadeIn(ctrl_lbl), run_time=0.5)
        self.play(
            AnimationGroup(
                FadeIn(vj_grp), FadeIn(vs_txt), FadeIn(vm_grp),
                lag_ratio=0.2,
            ),
            run_time=1.0,
        )
        # ── t=0:14  Deltas ─────────────────────────────────────────────────
        # Narration: "Under this equal footing, V-JEPA still wins consistently..."
        self.remove(_sub)
        _sub = make_sub("V-JEPA wins: +0.7 K400, +0.5 SSv2, +3.4 mAP AVA — even under identical conditions.")
        self.add(_sub)
        wait_to(self, t0, 14.0, "ch6_2 deltas")
        self.play(FadeIn(deltas, lag_ratio=0.3), run_time=1.1)

        # ── t=0:28  Clear and show efficiency chart ───────────────────────
        # Narration: "But the most striking result is about data efficiency."
        self.remove(_sub)
        _sub = make_sub("The most striking advantage: V-JEPA's extraordinary sample efficiency.")
        self.add(_sub)
        wait_to(self, t0, 28.0, "ch6_2 eff")
        self.play(FadeOut(VGroup(ctrl_lbl, boxes_row, deltas)), run_time=0.6)

        # ── Phase B: Sample Efficiency chart ─────────────────────────────
        eff_lbl = Text(
            "Pretraining samples seen  (millions)  — fewer is better",
            font_size=18, color=C_TEXT,
        ).next_to(hdr, DOWN, buff=0.45)
        self.play(FadeIn(eff_lbl), run_time=0.5)

        eff_vals  = [210, 1600, 1900, 39000]
        eff_names = ["V-JEPA", "VideoMAEv2", "DINOv2", "OpenCLIP"]
        eff_cols  = [C_ORANGE, "#9CA3AF", "#9CA3AF", "#9CA3AF"]
        eff_chart = BarChart(
            values=eff_vals, bar_names=eff_names,
            y_range=[0, 42000, 10000], x_length=9.0, y_length=3.2,
            bar_colors=eff_cols,
            x_axis_config={"font_size": 16, "color": C_TEXT},
            y_axis_config={"color": C_TEXT},
        ).set_z_index(1)
        eff_chart.next_to(eff_lbl, DOWN, buff=0.42)

        # ── t=0:32  Show efficiency chart ──────────────────────────────────
        # Narration: "V-JEPA achieves all these superior results after seeing just 210M..."
        self.remove(_sub)
        _sub = make_sub("V-JEPA uses only 210M pretraining frames — far fewer samples than any competitor.")
        self.add(_sub)
        wait_to(self, t0, 32.0, "ch6_2 chart")
        self.play(Create(eff_chart), run_time=1.6)

        eff_bar_lbls = eff_chart.get_bar_labels(
            font_size=13,
            label_constructor=lambda v: Text(f"{int(v):,}M", font_size=12),
        )
        # ── t=0:45  Bar labels + highlight ────────────────────────────────
        # Narration: "VideoMAEv2 needs 1,600 million."
        self.remove(_sub)
        _sub = make_sub("VideoMAEv2: 1,600M samples. DINOv2: 1,900M. OpenCLIP: 39,000M samples needed.")
        self.add(_sub)
        wait_to(self, t0, 45.0, "ch6_2 barlbls")
        self.play(FadeIn(eff_bar_lbls, lag_ratio=0.2), run_time=0.8)
        self.play(Indicate(eff_chart.bars[0], color=C_YELLOW, scale_factor=1.06), run_time=0.5)

        # ── t=0:57  Efficiency note ───────────────────────────────────────
        # Narration: "V-JEPA is not merely more accurate — it is dramatically more sample-efficient."
        self.remove(_sub)
        _sub = make_sub("V-JEPA: ~200x fewer samples than OpenCLIP, yet outperforms on every motion task.")
        self.add(_sub)
        wait_to(self, t0, 57.0, "ch6_2 note")
        eff_note = Text(
            "V-JEPA uses ~200× fewer samples than OpenCLIP  —  yet outperforms it on every motion task",
            font_size=15, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.42)
        self.play(FadeIn(eff_note, shift=UP * 0.2), run_time=0.7)

        all_62 = VGroup(hdr, eff_lbl, eff_chart, eff_bar_lbls, eff_note)
        self.remove(_sub)
        sync_wait(self, t0, dur, "ch6_2")
        self.play(FadeOut(all_62), run_time=0.8)


# ═════════════════════════════════════════════════════════════════════════
# CHAPTER 7 SCENE
# ═════════════════════════════════════════════════════════════════════════
class Ch7Scene(Scene):
    def construct(self):
        self.camera.background_color = BG
        self._ch7_attentive_probing()

    def _ch7_attentive_probing(self, is_master: bool = False):
        AUDIO = "media/ch7_new.mp3"
        dur = safe_sound(self, AUDIO, is_master=is_master)
        t0  = self.renderer.time

        hdr = chapter_header(7, "Frozen Evaluation: Attentive Probing")
        self.play(FadeIn(hdr, shift=DOWN * 0.2), run_time=0.8)

        # ── t=0:08  Phase 1: Frozen Encoder box ──────────────────────────
        # Narration: "The entire V-JEPA video encoder — with hundreds of millions of parameters..."
        wait_to(self, t0, 8.0, "ch7 frozen")
        _sub = make_sub("The V-JEPA encoder is completely frozen during evaluation — no backbone weight updates.")
        self.add(_sub)
        frozen_rect = rounded_box(3.3, 1.3, C_SILVER)
        frozen_line1 = Text("V-JEPA Encoder", font_size=18, color=C_SILVER, weight=BOLD)
        frozen_line2 = Text("[FROZEN — no parameter updates]", font_size=13, color=C_DIM)
        frozen_inner = VGroup(frozen_line1, frozen_line2).arrange(DOWN, buff=0.1)
        frozen_grp = VGroup(frozen_rect, frozen_inner)
        frozen_grp.move_to(LEFT * 4.0 + UP * 0.2)
        frozen_inner.move_to(frozen_rect)

        self.play(FadeIn(frozen_grp), run_time=0.8)

        # Lock badge — placed cleanly in top-right corner of the encoder box
        lock_badge = Text("FROZEN", font_size=12, color=C_RED, weight=BOLD)
        lock_bg = RoundedRectangle(
            width=1.1, height=0.3, corner_radius=0.06,
            fill_color=C_RED, fill_opacity=0.25,
            stroke_color=C_RED, stroke_width=1.2,
        )
        lock_badge.move_to(lock_bg)
        lock_grp = VGroup(lock_bg, lock_badge)
        lock_grp.next_to(frozen_rect, UP, buff=0.0).align_to(frozen_rect, RIGHT).set_z_index(5)
        self.play(FadeIn(lock_grp, scale=1.3), run_time=0.5)

        # ── Phase 2: Token strip (output of frozen encoder) ───────────────
        n_tok = 9
        tok_strip = VGroup(*[
            Square(side_length=0.27, fill_color=C_BLUE, fill_opacity=0.75, stroke_width=0)
            for _ in range(n_tok)
        ]).arrange(RIGHT, buff=0.07).set_z_index(2)
        tok_strip.move_to(ORIGIN + UP * 0.2)

        tok_lbl = Text("1568 spatial-temporal tokens", font_size=12, color=C_DIM)
        tok_lbl.next_to(tok_strip, DOWN, buff=0.12).set_z_index(2)

        arr_enc_tok = make_arrow(frozen_rect.get_right(), tok_strip.get_left(), color=C_SILVER)
        # ── t=0:14  Token strip ────────────────────────────────────────────
        # Narration: "Not a single backbone weight is updated during evaluation."
        self.remove(_sub)
        _sub = make_sub("Not a single backbone parameter is modified during downstream evaluation.")
        self.add(_sub)
        wait_to(self, t0, 14.0, "ch7 tokens")
        self.play(GrowArrow(arr_enc_tok), FadeIn(tok_strip), FadeIn(tok_lbl), run_time=1.0)

        # ── t=0:19  Phase 3: Cross-Attention Probe ────────────────────────
        # Narration: "Instead, a lightweight cross-attention probe is trained on top."
        self.remove(_sub)
        _sub = make_sub("A cross-attention probe is the only trainable component on top.")
        self.add(_sub)
        wait_to(self, t0, 19.0, "ch7 probe")
        probe_rect = rounded_box(2.9, 1.2, C_GREEN)
        probe_line1 = Text("Cross-Attention Probe", font_size=16, color=C_GREEN, weight=BOLD)
        probe_line2 = Text("(trainable)", font_size=13, color=C_DIM)
        probe_inner = VGroup(probe_line1, probe_line2).arrange(DOWN, buff=0.08)
        probe_grp = VGroup(probe_rect, probe_inner)
        probe_grp.move_to(RIGHT * 3.6 + UP * 0.6)
        probe_inner.move_to(probe_rect)

        arr_tok_probe = make_arrow(tok_strip.get_right(), probe_rect.get_left(), color=C_GREEN)
        self.play(GrowArrow(arr_tok_probe), FadeIn(probe_grp), run_time=1.0)

        # ── t=0:23  Query token + attention arrows ────────────────────────
        # Narration: "A single learnable query token attends across all 1568 spatial-temporal tokens..."
        self.remove(_sub)
        _sub = make_sub("One query token attends all 1,568 spatial-temporal tokens via cross-attention.")
        self.add(_sub)
        wait_to(self, t0, 23.0, "ch7 query")
        # Query token (from below)
        query_circ = Circle(radius=0.28, fill_color=C_YELLOW, fill_opacity=0.9, stroke_width=0)
        query_lbl_txt = Text("Query  q", font_size=13, color=C_YELLOW)
        query_grp = VGroup(query_circ, query_lbl_txt).arrange(DOWN, buff=0.08)
        query_grp.move_to(RIGHT * 3.6 + DOWN * 1.1)

        arr_q = Arrow(
            query_circ.get_top(), probe_rect.get_bottom(),
            color=C_YELLOW, buff=0.06, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.14,
        ).set_z_index(1)
        self.play(FadeIn(query_grp, shift=UP * 0.25), GrowArrow(arr_q), run_time=0.8)

        # Simulate attention — highlight 3 tokens brighter
        hi_toks = VGroup(*[
            tok_strip[i].copy().set_fill(C_YELLOW, opacity=0.95).set_z_index(3)
            for i in [1, 4, 7]
        ])
        self.play(FadeIn(hi_toks, lag_ratio=0.25), run_time=0.6)
        self.wait(0.3)

        # Attention weight arrows (5 representative arrows, varying opacity)
        attn_start_pts = [tok_strip[i].get_top() for i in [0, 2, 4, 6, 8]]
        probe_bot = probe_rect.get_bottom()
        spread    = [LEFT * 0.9, LEFT * 0.45, ORIGIN, RIGHT * 0.45, RIGHT * 0.9]
        attn_arrows = VGroup(*[
            Arrow(
                attn_start_pts[i],
                probe_bot + spread[i],
                color=C_YELLOW,
                buff=0.05,
                stroke_width=0.8 + i * 0.4,
                max_tip_length_to_length_ratio=0.15,
            ).set_z_index(1).set_stroke(opacity=0.25 + i * 0.15)
            for i in range(5)
        ])
        self.play(
            AnimationGroup(*[GrowArrow(a) for a in attn_arrows], lag_ratio=0.1),
            run_time=1.0,
        )

        # ── Phase 4: Pooled repr + Linear Classifier ─────────────────────
        pool_dot = Dot(radius=0.30, fill_color=C_PURPLE, fill_opacity=0.95).set_z_index(4)
        pool_lbl = Text("Pooled repr.", font_size=13, color=C_PURPLE).set_z_index(5)
        # Label above dot — arrow arrives from below (probe_rect.get_top()), so no overlap
        pool_grp = VGroup(pool_lbl, pool_dot).arrange(DOWN, buff=0.1)
        pool_grp.move_to(RIGHT * 3.6 + UP * 2.1)

        lin_rect = RoundedRectangle(
            width=2.1, height=0.78, corner_radius=0.1,
            fill_color="#7C3AED", fill_opacity=0.28,
            stroke_color="#7C3AED", stroke_width=2.0,
        ).set_z_index(2)
        lin_lbl = Text("Linear\nClassifier", font_size=14, color="#A78BFA")
        lin_lbl.move_to(lin_rect).set_z_index(3)
        lin_grp = VGroup(lin_rect, lin_lbl)
        lin_grp.move_to(RIGHT * 5.6 + UP * 2.1)
        lin_lbl.move_to(lin_rect)

        arr_p_pool = make_arrow(probe_rect.get_top(), pool_dot.get_bottom(), color=C_GREEN)
        arr_pool_lin = make_arrow(pool_dot.get_right(), lin_rect.get_left(), color=C_PURPLE)

        # ── t=0:33  Pooled repr ────────────────────────────────────────────
        # Narration: "This pooled vector then feeds a simple linear classifier."
        self.remove(_sub)
        _sub = make_sub("The pooled vector feeds a linear classifier, testing pure representation quality.")
        self.add(_sub)
        wait_to(self, t0, 33.0, "ch7 pool")
        self.play(GrowArrow(arr_p_pool), FadeIn(pool_grp), run_time=0.7)
        self.play(GrowArrow(arr_pool_lin), FadeIn(lin_grp), run_time=0.7)

        # ── t=0:37  Attention Pooling formula ─────────────────────────────
        # Narration: "The attentive pooling formula is..."
        self.remove(_sub)
        _sub = make_sub("Attentive pooling: weighted sum over token values using learned cross-attention scores.")
        self.add(_sub)
        wait_to(self, t0, 37.0, "ch7 formula")
        # Attention Pooling formula
        ap_eq = MathTex(
            r"\mathrm{AP}(q,S) = \sum_i "
            r"\frac{e^{\,q^\top W_k s_i}}{\sum_j e^{\,q^\top W_k s_j}}\, W_v s_i",
            font_size=22,
        ).set_color(C_GREEN).to_edge(DOWN, buff=0.5)
        self.play(Write(ap_eq), run_time=1.3)
        self.wait(0.5)

        # Cleanup Phase 1-4
        probe_scene = VGroup(
            frozen_grp, lock_grp, arr_enc_tok, tok_strip, tok_lbl,
            arr_tok_probe, probe_grp, query_grp, arr_q, hi_toks, attn_arrows,
            arr_p_pool, pool_grp, arr_pool_lin, lin_grp, ap_eq,
        )
        self.play(FadeOut(probe_scene), run_time=0.8)

        # ── t=0:47  Phase 5: Multi-task downstream results ────────────────
        # Narration: "Using this single frozen encoder, V-JEPA achieves 77.9 percent on ImageNet-1K..."
        self.remove(_sub)
        _sub = make_sub("One frozen encoder: 82.1% K400, 71.2% SSv2, 77.9% ImageNet, 33.7 mAP AVA, 60.3% Places.")
        self.add(_sub)
        wait_to(self, t0, 47.0, "ch7 results")
        res_title = Text(
            "One Frozen Encoder  —  Multiple Downstream Tasks",
            font_size=24, color=C_YELLOW, weight=BOLD,
        ).next_to(hdr, DOWN, buff=0.48)
        self.play(FadeIn(res_title), run_time=0.6)

        task_data = [
            ("Kinetics-400    Action Recognition",  "82.1%",    C_ORANGE),
            ("SSv2            Physical Motion",      "71.2%",    C_BLUE),
            ("ImageNet-1K     Image Classification", "77.9%",    C_GREEN),
            ("AVA             Action Localization",  "33.7 mAP", C_PURPLE),
            ("Places205       Scene Recognition",    "60.3%",    C_TEAL),
        ]
        task_rows = VGroup()
        for name, score, col in task_data:
            bg = RoundedRectangle(
                width=7.8, height=0.62, corner_radius=0.1,
                fill_color=col, fill_opacity=0.12,
                stroke_color=col, stroke_width=1.3,
            ).set_z_index(1)
            n_txt = Text(name, font_size=15, color=C_TEXT).move_to(bg).shift(LEFT * 2.1).set_z_index(2)
            s_txt = Text(score, font_size=18, color=col, weight=BOLD).move_to(bg).shift(RIGHT * 2.9).set_z_index(2)
            task_rows.add(VGroup(bg, n_txt, s_txt))
        task_rows.arrange(DOWN, buff=0.2).next_to(res_title, DOWN, buff=0.4)

        self.play(FadeIn(task_rows, lag_ratio=0.28), run_time=1.8)
        self.wait(0.5)
        # NOTE: keep results visible — don’t fade out here.
        # Results stay on screen until right before conclusion bullets (t=67s).

        # ── t=1:07  Phase 6: Conclusion bullets ───────────────────────────
        # Narration: "V-JEPA has proven the ultimate thesis of self-supervised representation learning..."
        self.remove(_sub)
        _sub = make_sub("V-JEPA: predict in latent space -> learn rich, transferable video representations.")
        self.add(_sub)
        wait_to(self, t0, 67.0, "ch7 conclusion")
        # Fade out results just before showing conclusion
        self.play(FadeOut(VGroup(hdr, res_title, task_rows)), run_time=0.7)
        concl_title = Text(
            "V-JEPA  —  Key Take-aways",
            font_size=34, color=C_YELLOW, weight=BOLD,
        ).to_edge(UP, buff=0.48)
        self.play(FadeIn(concl_title, shift=DOWN * 0.2), run_time=0.7)

        bullets = VGroup(
            Text("1.  Predict in latent space  —  not pixels",
                 font_size=21, color=C_TEXT),
            Text("2.  3D Multi-Block masking prevents temporal information shortcuts",
                 font_size=21, color=C_TEXT),
            Text("3.  Stop-gradient + EMA  =  stable, collapse-free training",
                 font_size=21, color=C_TEXT),
            Text("4.  Frozen evaluation reveals true representation quality",
                 font_size=21, color=C_TEXT),
            Text("5.  210M samples  —  ~200x more efficient than OpenCLIP",
                 font_size=21, color=C_YELLOW),
        ).arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        bullets.next_to(concl_title, DOWN, buff=0.5)

        self.play(FadeIn(bullets, lag_ratio=0.42), run_time=2.5)
        self.play(Indicate(bullets[-1], color=C_YELLOW, scale_factor=1.03), run_time=0.6)

        all_ch7 = VGroup(concl_title, bullets)
        self.remove(_sub)
        sync_wait(self, t0, dur, "ch7")
        self.play(FadeOut(all_ch7), run_time=0.8)

        # ── End card ──────────────────────────────────────────────────────
        v_big  = Text("V-JEPA", font_size=72, color=C_YELLOW, weight=BOLD).center()
        sub1   = Text("Video Joint Embedding Predictive Architecture", font_size=22, color=C_TEXT)
        sub1.next_to(v_big, DOWN, buff=0.32)
        sub2   = Text("Meta AI Research  ·  ICLR 2024", font_size=16, color=C_DIM)
        sub2.next_to(sub1, DOWN, buff=0.22)
        end_grp = VGroup(v_big, sub1, sub2)

        self.play(FadeIn(end_grp, lag_ratio=0.35), run_time=2.0)
        self.wait(3.5)
        self.play(FadeOut(end_grp), run_time=1.0)


# ═════════════════════════════════════════════════════════════════════════
# COMBINED SCENE — Chapters 5 + 6 + 7 in one render
# ═════════════════════════════════════════════════════════════════════════
class VJEPACh567Scene(Ch5Scene, Ch6Scene, Ch7Scene):
    """
    Renders Chapters 5, 6, 7 as one continuous video by inheriting and
    calling all chapter methods in sequence.

    Render:
        manim -ql vjepa_ch567_full.py VJEPACh567Scene
    """
    def construct(self):
        self.camera.background_color = BG

        # Audio: each chapter section adds its own track at the right moment

        # Chapter 5 — Loss, Stop-Gradient, Collapse, EMA
        self._ch5_1_loss_stopgrad(is_master=False)
        self.wait(0.8)
        self._ch5_2_collapse_ema(is_master=False)
        self.wait(0.8)

        # Chapter 6 — Benchmarks, Sample Efficiency
        self._ch6_1_benchmarks(is_master=False)
        self.wait(0.8)
        self._ch6_2_efficiency(is_master=False)
        self.wait(0.8)

        # Chapter 7 — Attentive Probing, Multi-task Results, Conclusion
        self._ch7_attentive_probing(is_master=False)


if __name__ == "__main__":
    import sys
    import subprocess
    cmd = [sys.executable, "-m", "manim", "-ql", __file__, "VJEPACh567Scene"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

