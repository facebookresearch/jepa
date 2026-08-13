import os
# Khai báo đường dẫn MiKTeX để đảm bảo tương thích hệ thống
os.environ["PATH"] += os.pathsep + r"C:\Users\tony\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

from manim import *

class Chapter6Scene(Scene):
    import os
# Khai báo đường dẫn MiKTeX để đảm bảo tương thích hệ thống
os.environ["PATH"] += os.pathsep + r"C:\Users\tony\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

from manim import *

class Chapter6Scene(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"
        
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

        # 3. Hiệu ứng Highlight dòng của V-JEPA (Sử dụng Z-Index để nổi bật)
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
        self.wait(3.0)

        # 4. Dọn dẹp màn hình
        ch6_group = VGroup(title, comparison_table, highlight_rect, ssv2_highlight)
        self.play(FadeOut(ch6_group))

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

        # 3. Hiệu ứng Highlight dòng của V-JEPA (Sử dụng Z-Index để nổi bật)
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
        self.wait(3.0)

        # 4. Dọn dẹp màn hình
        ch6_group = VGroup(title, comparison_table, highlight_rect, ssv2_highlight)
        self.play(FadeOut(ch6_group))