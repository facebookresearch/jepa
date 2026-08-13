import os
os.environ["PATH"] += os.pathsep + r"C:\Users\tony\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

from manim import *
from mutagen.mp3 import MP3
import numpy as np


def audio_duration(path: str) -> float:
    """Trả về thời lượng thực tế (giây) của file mp3."""
    return MP3(path).info.length


def sync_wait(scene: Scene, start_time: float, duration: float, label: str = ""):
    """
    Tự động chờ thêm cho đủ khớp với thời lượng audio.
    start_time: thời điểm (scene.renderer.time) lúc bắt đầu phát audio.
    duration:   thời lượng thực tế của file mp3 (giây).
    Nếu animation đã chạy lâu hơn audio -> không chờ thêm (in cảnh báo ra console).
    Nếu animation ngắn hơn audio -> tự thêm self.wait() cho khớp.
    """
    elapsed = scene.renderer.time - start_time
    remaining = duration - elapsed
    if remaining > 0:
        scene.wait(remaining)
    else:
        print(f"⚠️  [{label}] Animation ({elapsed:.1f}s) đã dài hơn audio ({duration:.1f}s), "
              f"vượt {abs(remaining):.1f}s — audio sẽ kết thúc trước khi animation xong.")


class Chapter5Scene(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"
        self.part_5_1_loss_and_stop_gradient()
        self.part_5_2_representation_collapse_and_ema()

    def part_5_1_loss_and_stop_gradient(self):
        audio_path = "media/ch5_part1.mp3"
        duration = audio_duration(audio_path)
        self.add_sound(audio_path)
        start_time = self.renderer.time

        loss_formula = MathTex(
            r"\mathcal{L}_{\text{V-JEPA}} = \frac{1}{M} \sum_{k \in \{i_1, \dots, i_M\}} \|\hat{s}_k - s_k\|_1"
        ).scale(1.1).to_edge(UP).set_z_index(5)

        self.play(Write(loss_formula), run_time=2.0)
        self.wait(1.0)

        predictor = Rectangle(width=2, height=1, color=ORANGE).shift(RIGHT * 3 + DOWN * 1).set_fill(ORANGE, 0.2).set_z_index(1)
        pred_text = Text("Predictor", font_size=20).move_to(predictor).set_z_index(2)

        ctx_enc = Rectangle(width=2.5, height=1, color=BLUE).shift(LEFT * 3 + UP * 0.5).set_fill(BLUE, 0.2).set_z_index(1)
        ctx_text = Text("Context Encoder", font_size=20).move_to(ctx_enc).set_z_index(2)

        tgt_enc = Rectangle(width=2.5, height=1, color=GRAY).shift(LEFT * 3 + DOWN * 2.5).set_fill(GRAY, 0.2).set_z_index(1)
        tgt_text = Text("Target Encoder", font_size=20).move_to(tgt_enc).set_z_index(2)

        network_group = VGroup(predictor, pred_text, ctx_enc, ctx_text, tgt_enc, tgt_text)
        self.play(FadeIn(network_group, lag_ratio=0.2), run_time=1.5)

        grad_pred = Arrow(loss_formula.get_bottom(), predictor.get_top(), color=YELLOW, buff=0.2).set_z_index(0)
        grad_ctx = Arrow(predictor.get_left(), ctx_enc.get_right(), color=YELLOW, buff=0.2).set_z_index(0)

        grad_tgt = Arrow(predictor.get_bottom(), tgt_enc.get_right() + RIGHT * 1.5, color=YELLOW, buff=0.2).set_z_index(0)
        stop_wall = Line(UP, DOWN, color=RED, stroke_width=8).scale(0.6).move_to(tgt_enc.get_right() + RIGHT * 0.5).set_z_index(3)
        stop_text = Text("stop-gradient", font_size=20, color=RED).next_to(stop_wall, RIGHT).set_z_index(3)

        self.play(Create(grad_pred), run_time=1.0)
        self.play(Create(grad_ctx), run_time=1.0)
        self.play(Create(grad_tgt), run_time=1.0)

        self.play(
            FadeIn(stop_wall, scale=1.5),
            FadeIn(stop_text, shift=LEFT),
            grad_tgt.animate.put_start_and_end_on(grad_tgt.get_start(), stop_wall.get_center() + RIGHT * 0.1),
            run_time=1.0, rate_func=smooth
        )

        # Tự động chờ thêm cho khớp với độ dài audio thật (thay vì wait(3.5) cố định)
        sync_wait(self, start_time, duration, label="ch5_part1")

        ch5_part1_group = VGroup(loss_formula, network_group, grad_pred, grad_ctx, grad_tgt, stop_wall, stop_text)
        self.play(FadeOut(ch5_part1_group))

    def part_5_2_representation_collapse_and_ema(self):
        audio_path = "media/ch5_part2.mp3"
        duration = audio_duration(audio_path)
        self.add_sound(audio_path)
        start_time = self.renderer.time

        title = Text("Representation Collapse", font_size=40, color=RED).to_edge(UP)
        self.play(FadeIn(title))

        scattered_dots = VGroup(*[
            Dot(radius=0.1, color=color).move_to(np.random.uniform(-4, 4, 3) * [1, 0.5, 0])
            for color in [BLUE, GREEN, YELLOW, PURPLE, TEAL, MAROON] for _ in range(15)
        ]).set_z_index(1)

        self.play(FadeIn(scattered_dots, lag_ratio=0.01), run_time=1.5)
        self.wait(0.5)

        collapse_dot = Dot(radius=0.3, color=RED).move_to(ORIGIN).set_z_index(2)
        trivial_text = Text("Trivial Solution", font_size=28, color=RED).next_to(collapse_dot, DOWN).set_z_index(2)

        self.play(
            ReplacementTransform(scattered_dots, collapse_dot),
            FadeIn(trivial_text, shift=UP),
            run_time=1.5, rate_func=smooth
        )
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, collapse_dot, trivial_text)))

        ema_title = Text("Exponential Moving Average (EMA)", font_size=36, color=BLUE).to_edge(UP)
        ema_formula = MathTex(
            r"\bar{\theta}_t \leftarrow m \cdot \bar{\theta}_{t-1} + (1 - m) \cdot \theta_t"
        ).scale(1.2).shift(UP * 1)

        self.play(FadeIn(ema_title), Write(ema_formula), run_time=1.5)

        mad_title = Text("When the Predictor is optimal under L1 loss:", font="Arial", font_size=24, color=WHITE).shift(DOWN * 0.5)
        opt_formula = MathTex(
            r"p^\star(z_N(\theta)) = \text{median}(X | z_N(\theta))"
        ).scale(0.9).next_to(mad_title, DOWN, buff=0.3)

        grad_formula = MathTex(
            r"\nabla_{\theta} \mathbb{E}\|p^\star(z_N(\theta)) - X\|_1 = \nabla_{\theta} \sum_{l=1}^d \text{MAD}(X_l | z_N(\theta))"
        ).scale(0.9).next_to(opt_formula, DOWN, buff=0.5)

        self.play(FadeIn(mad_title), Write(opt_formula), run_time=1.5)
        self.play(Write(grad_formula), run_time=2.0)

        # Tự động chờ thêm cho khớp với độ dài audio thật (thay vì wait(4.0) cố định)
        sync_wait(self, start_time, duration, label="ch5_part2")

        ch5_part2_group = VGroup(ema_title, ema_formula, mad_title, opt_formula, grad_formula)
        self.play(FadeOut(ch5_part2_group))


class Chapter6Scene(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"

        audio_path = "media/ch6_audio.mp3"
        duration = audio_duration(audio_path)
        self.add_sound(audio_path)
        start_time = self.renderer.time

        # 1. Tiêu đề Phân cảnh
        title = Text("V-JEPA vs. Competitors", font="Arial", font_size=40, color=BLUE).to_edge(UP)
        self.play(FadeIn(title, shift=UP), run_time=1.5)

        # 2. Dữ liệu bảng so sánh hiệu năng
        table_data = [
            ["OpenCLIP", "ViT-G/14", "83.3%", "39.0%"],
            ["DINOv2", "ViT-g/14", "84.4%", "50.0%"],
            ["VideoMAE", "ViT-L/16", "77.9%", "61.2%"],
            ["V-JEPA", "ViT-H/16", "82.1%", "71.2%"]
        ]

        comparison_table = Table(
            table_data,
            col_labels=[
                Text("Algorithm", font="Arial", font_size=24),
                Text("Architecture", font="Arial", font_size=24),
                Text("K400", font="Arial", font_size=24),
                Text("SSv2", font="Arial", font_size=24)
            ],
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": GRAY}
        ).scale(0.7).next_to(title, DOWN, buff=1.0).set_z_index(0)

        self.play(DrawBorderThenFill(comparison_table), run_time=2.0)
        self.wait(1.0)

        # 3. Hiệu ứng Highlight dòng của V-JEPA
        highlight_box = comparison_table.get_rows()[4].copy()
        highlight_box.set_color(YELLOW).set_z_index(1)

        highlight_rect = SurroundingRectangle(
            comparison_table.get_rows()[4],
            color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.2,
            buff=0.1
        ).set_z_index(0)

        ssv2_highlight = Text("+10% vs VideoMAE on SSv2", font="Arial", font_size=28, color=YELLOW)
        ssv2_highlight.next_to(comparison_table, DOWN, buff=0.8).set_z_index(2)

        self.play(
            Create(highlight_rect),
            Transform(comparison_table.get_rows()[4], highlight_box),
            run_time=1.0
        )
        self.play(FadeIn(ssv2_highlight, shift=UP), run_time=1.0)

        # Tự động chờ thêm cho khớp với độ dài audio thật (thay vì wait(3.0) cố định)
        sync_wait(self, start_time, duration, label="ch6_audio")

        # 4. Dọn dẹp màn hình
        ch6_group = VGroup(title, comparison_table, highlight_rect, ssv2_highlight)
        self.play(FadeOut(ch6_group))


class Chapter7Scene(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"

        audio_path = "media/ch7_audio.mp3"
        duration = audio_duration(audio_path)
        self.add_sound(audio_path)
        start_time = self.renderer.time

        # 1. Tiêu đề Attentive Probing
        title = Text("Attentive Probing", font="Arial", font_size=40, color=GREEN).to_edge(UP)
        self.play(FadeIn(title, shift=UP), run_time=1.5)

        frozen_box = Rectangle(width=3, height=1.5, color=GRAY).shift(LEFT * 3).set_fill(GRAY, 0.3)
        frozen_text = Text("Frozen V-JEPA\nEncoder", font="Arial", font_size=20).move_to(frozen_box)

        probe_box = Rectangle(width=2.5, height=1.5, color=GREEN).shift(RIGHT * 2).set_fill(GREEN, 0.3)
        probe_text = Text("Attention\nProbe", font="Arial", font_size=20).move_to(probe_box)

        arrow = Arrow(frozen_box.get_right(), probe_box.get_left(), color=YELLOW)

        group_probe = VGroup(frozen_box, frozen_text, probe_box, probe_text, arrow)
        self.play(FadeIn(group_probe, lag_ratio=0.3), run_time=2.0)
        self.wait(3.0)

        self.play(FadeOut(group_probe), FadeOut(title))

        # 2. Phần Kết luận (Conclusion)
        conclusion_title = Text("Conclusion", font="Arial", font_size=40, color=BLUE).to_edge(UP)

        bullet1 = Text("• Spatial-Temporal Masking", font="Arial", font_size=24, color=WHITE)
        bullet2 = Text("• Stop-Gradient Mechanism", font="Arial", font_size=24, color=WHITE)
        bullet3 = Text("• Exponential Moving Average (EMA)", font="Arial", font_size=24, color=WHITE)

        conclusion_list = VGroup(bullet1, bullet2, bullet3).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(conclusion_title, DOWN, buff=1.0)

        self.play(FadeIn(conclusion_title, shift=UP), run_time=1.0)
        self.play(FadeIn(conclusion_list, lag_ratio=0.5), run_time=3.0)

        # Tự động chờ thêm cho khớp với độ dài audio thật (thay vì wait(4.0) cố định)
        sync_wait(self, start_time, duration, label="ch7_audio")

        self.play(FadeOut(conclusion_title), FadeOut(conclusion_list))