import matplotlib.pyplot as plt
from PIL import Image

def plot_guess_img(input_tensor, output_filename, reference_img='evals/video_classification_frozen/reference_with_border.png', scale=150):
    assert input_tensor.numel() == 165, "input tensor must have exactly 165 entries"
    background = Image.open(reference_img)

    plt.figure(figsize=(background.width / 100, background.height / 100), dpi=100)
    plt.imshow(background)
    plt.axis('off')

    start_x, start_y = 520, 228
    spacing = 63
    scatter_x = []
    scatter_y = []
    sizes = []

    for idx in range(input_tensor.numel()):
        row = idx // 15
        col = idx % 15
        x = start_x + col * spacing
        y = start_y + row * spacing
        scatter_x.append(x)
        scatter_y.append(y)
        sizes.append(input_tensor[idx].item() * scale)

    plt.scatter(scatter_x, scatter_y, s=sizes, c='green', alpha=0.4)
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
