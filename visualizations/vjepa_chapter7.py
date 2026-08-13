import os
os.environ["PATH"] += os.pathsep + r"C:\Users\tony\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

from manim import *

class Chapter7Scene(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"
        self.add_sound("media/ch7_audio.mp3")
        
        # 1. Tiêu đề Attentive Probing
        title = Text("Attentive Probing", font="Arial", font_size=40, color=GREEN).to_edge(UP)
        self.play(FadeIn(title, shift=UP), run_time=1.5)

        # Mô tả cơ chế (Frozen Encoder vs Lightweight Probe)
        frozen_box = Rectangle(width=3, height=1.5, color=GRAY).shift(LEFT*3).set_fill(GRAY, 0.3)
        frozen_text = Text("Frozen V-JEPA\nEncoder", font="Arial", font_size=20).move_to(frozen_box)
        
        probe_box = Rectangle(width=2.5, height=1.5, color=GREEN).shift(RIGHT*2).set_fill(GREEN, 0.3)
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
        self.wait(4.0)

        self.play(FadeOut(conclusion_title), FadeOut(conclusion_list))