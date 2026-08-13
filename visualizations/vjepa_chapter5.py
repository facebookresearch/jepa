import os
os.environ["PATH"] += os.pathsep + r"C:\Users\tony\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

from manim import *
import numpy as np

class Chapter5Scene(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"
        self.part_5_1_loss_and_stop_gradient()
        self.part_5_2_representation_collapse_and_ema()

    def part_5_1_loss_and_stop_gradient(self):
        # Phát âm thanh đoạn 1 (Đảm bảo file tồn tại ở đường dẫn này)
        self.add_sound("media/ch5_part1.mp3")
        
        loss_formula = MathTex(
            r"\mathcal{L}_{\text{V-JEPA}} = \frac{1}{M} \sum_{k \in \{i_1, \dots, i_M\}} \|\hat{s}_k - s_k\|_1"
        ).scale(1.1).to_edge(UP).set_z_index(5)
        
        self.play(Write(loss_formula), run_time=2.0)
        self.wait(1.0)

        predictor = Rectangle(width=2, height=1, color=ORANGE).shift(RIGHT*3 + DOWN*1).set_fill(ORANGE, 0.2).set_z_index(1)
        pred_text = Text("Predictor", font_size=20).move_to(predictor).set_z_index(2)
        
        ctx_enc = Rectangle(width=2.5, height=1, color=BLUE).shift(LEFT*3 + UP*0.5).set_fill(BLUE, 0.2).set_z_index(1)
        ctx_text = Text("Context Encoder", font_size=20).move_to(ctx_enc).set_z_index(2)
        
        tgt_enc = Rectangle(width=2.5, height=1, color=GRAY).shift(LEFT*3 + DOWN*2.5).set_fill(GRAY, 0.2).set_z_index(1)
        tgt_text = Text("Target Encoder", font_size=20).move_to(tgt_enc).set_z_index(2)

        network_group = VGroup(predictor, pred_text, ctx_enc, ctx_text, tgt_enc, tgt_text)
        self.play(FadeIn(network_group, lag_ratio=0.2), run_time=1.5)

        grad_pred = Arrow(loss_formula.get_bottom(), predictor.get_top(), color=YELLOW, buff=0.2).set_z_index(0)
        grad_ctx = Arrow(predictor.get_left(), ctx_enc.get_right(), color=YELLOW, buff=0.2).set_z_index(0)
        
        grad_tgt = Arrow(predictor.get_bottom(), tgt_enc.get_right() + RIGHT*1.5, color=YELLOW, buff=0.2).set_z_index(0)
        stop_wall = Line(UP, DOWN, color=RED, stroke_width=8).scale(0.6).move_to(tgt_enc.get_right() + RIGHT*0.5).set_z_index(3)
        stop_text = Text("stop-gradient", font_size=20, color=RED).next_to(stop_wall, RIGHT).set_z_index(3)

        self.play(Create(grad_pred), run_time=1.0)
        self.play(Create(grad_ctx), run_time=1.0)
        self.play(Create(grad_tgt), run_time=1.0)
        
        self.play(
            FadeIn(stop_wall, scale=1.5),
            FadeIn(stop_text, shift=LEFT),
            grad_tgt.animate.put_start_and_end_on(grad_tgt.get_start(), stop_wall.get_center() + RIGHT*0.1),
            run_time=1.0, rate_func=smooth
        )
        # Giữ cảnh lâu hơn một chút để đợi giọng đọc TTS kết thúc
        self.wait(3.5)

        ch5_part1_group = VGroup(loss_formula, network_group, grad_pred, grad_ctx, grad_tgt, stop_wall, stop_text)
        self.play(FadeOut(ch5_part1_group))

    def part_5_2_representation_collapse_and_ema(self):
        # Phát âm thanh đoạn 2
        self.add_sound("media/ch5_part2.mp3")
        
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
        ).scale(1.2).shift(UP*1)
        
        self.play(FadeIn(ema_title), Write(ema_formula), run_time=1.5)

        mad_title = Text("When the Predictor is optimal under L1 loss:", font="Arial", font_size=24, color=WHITE).shift(DOWN*0.5)
        opt_formula = MathTex(
            r"p^\star(z_N(\theta)) = \text{median}(X | z_N(\theta))"
        ).scale(0.9).next_to(mad_title, DOWN, buff=0.3)

        grad_formula = MathTex(
            r"\nabla_{\theta} \mathbb{E}\|p^\star(z_N(\theta)) - X\|_1 = \nabla_{\theta} \sum_{l=1}^d \text{MAD}(X_l | z_N(\theta))"
        ).scale(0.9).next_to(opt_formula, DOWN, buff=0.5)

        self.play(FadeIn(mad_title), Write(opt_formula), run_time=1.5)
        self.play(Write(grad_formula), run_time=2.0)
        
        # Giữ cảnh lâu hơn để kết thúc trọn vẹn câu thoại
        self.wait(4.0)

        ch5_part2_group = VGroup(ema_title, ema_formula, mad_title, opt_formula, grad_formula)
        self.play(FadeOut(ch5_part2_group))