import ssl
import asyncio
import os


def patched_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False  # phải tắt trước, nếu không set verify_mode=CERT_NONE sẽ báo lỗi
    ctx.verify_mode = ssl.CERT_NONE  # Bỏ qua xác thực nghiêm ngặt để tránh lỗi chứng chỉ trên Windows
    return ctx


# QUAN TRỌNG: phải patch trước khi import edge_tts/aiohttp,
# vì aiohttp gọi ssl.create_default_context() ngay lúc import module.
ssl.create_default_context = patched_default_context
# ------------------------------------------------

import edge_tts
from mutagen.mp3 import MP3

VOICE = "en-GB-RyanNeural"
os.makedirs("media", exist_ok=True)

# Kịch bản các chương
TEXT_1 = "The V-JEPA loss function computes the L1 distance between the predicted features and the target features. To train the network effectively, a stop-gradient is applied to the target encoder, preventing the backward flow of gradients."

TEXT_2 = "In self-supervised learning, there is a dangerous system flaw called Representation Collapse, where the model outputs a trivial constant solution. V-JEPA mathematically prevents this by updating the target encoder using an Exponential Moving Average. This forces the model to constantly learn rich representations without ever collapsing."

TEXT_CH6 = "To see the superiority of V-JEPA, let's compare it with other major algorithm families. On the Something-Something-v2 dataset, which requires deep understanding of physical motion, V-JEPA achieves 71.2 percent accuracy, completely outperforming VideoMAE, and large image models like DINOv2 and OpenCLIP."

TEXT_CH7 = "Finally, let's look at Attentive Probing. Instead of freezing or fine-tuning the entire network, V-JEPA freezes the representation and trains a lightweight attention-based probe on top. This efficiently extracts task-specific information. In conclusion, V-JEPA sets a new standard for self-supervised video representation learning through spatial-temporal masking, stop-gradients, and exponential moving averages."


async def amain():
    print("Đang tiến hành tạo toàn bộ âm thanh TTS...")

    files = [
        (TEXT_1, "media/ch5_part1.mp3"),
        (TEXT_2, "media/ch5_part2.mp3"),
        (TEXT_CH6, "media/ch6_audio.mp3"),
        (TEXT_CH7, "media/ch7_audio.mp3"),
    ]

    for text, path in files:
        await edge_tts.Communicate(text, VOICE).save(path)
        duration = MP3(path).info.length
        print(f"✅ Đã tạo xong: {path} — {duration:.1f}s")

    print("🎉 Hoàn tất sinh tất cả các file âm thanh!")


if __name__ == "__main__":
    asyncio.run(amain())
