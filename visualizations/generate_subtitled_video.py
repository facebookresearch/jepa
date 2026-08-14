"""
Automated Pipeline: Generate V-JEPA Video with Exact Sentence-Synced Subtitles
-----------------------------------------------------------------------------
This script runs the complete end-to-end process:
1. Generates TTS audio (Male UK voice)
2. Builds master audio track (media/ch567_master_audio.wav)
3. Generates millisecond-accurate sentence subtitles (.srt and .vtt)
4. Renders Manim Community Edition scene (VJEPACh567Scene)
5. Hardcodes/burns subtitles into MP4 via FFmpeg

Run:
    python visualizations/generate_subtitled_video.py
"""

import os
import sys
import ssl
import asyncio
import subprocess
from pathlib import Path
from datetime import timedelta

# Ensure PATH has MiKTeX for LaTeX rendering
os.environ["PATH"] += os.pathsep + r"C:\Users\tony\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

# SSL patch for edge-tts
def patched_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = patched_default_context

import edge_tts
from pydub import AudioSegment

# Import TTS script texts
from visualizations.tts_ch567_full import CH5_1, CH5_2, CH6_1, CH6_2, CH7, VOICE

# Project root setup
ROOT_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = ROOT_DIR / "media"
VIS_DIR = ROOT_DIR / "visualizations"
MEDIA_DIR.mkdir(exist_ok=True)

AUDIO_FILES_MAP = [
    (CH5_1, MEDIA_DIR / "ch5_1_new.mp3"),
    (CH5_2, MEDIA_DIR / "ch5_2_new.mp3"),
    (CH6_1, MEDIA_DIR / "ch6_1_new.mp3"),
    (CH6_2, MEDIA_DIR / "ch6_2_new.mp3"),
    (CH7,   MEDIA_DIR / "ch7_new.mp3"),
]

FINAL_OUTPUT_MP4 = VIS_DIR / "vjepa_ch567_final_with_subtitles.mp4"
PERFECT_SRT = VIS_DIR / "vjepa_ch567_subtitles_perfect.srt"
PERFECT_VTT = VIS_DIR / "vjepa_ch567_subtitles_perfect.vtt"
MASTER_WAV = MEDIA_DIR / "ch567_master_audio.wav"


def format_srt_timestamp(ms: float) -> str:
    td = timedelta(milliseconds=ms)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


async def generate_tts_and_subtitles():
    print("Step 1: Generating TTS Audio & Sentence Boundaries...")
    segments = []
    texts = [CH5_1, CH5_2, CH6_1, CH6_2, CH7]
    
    for text, path in AUDIO_FILES_MAP:
        if not path.exists():
            print(f"  Generating {path.name}...")
            await edge_tts.Communicate(text, VOICE).save(str(path))
        seg = AudioSegment.from_file(str(path))
        segments.append(seg)

    print("Step 2: Building Master Audio Track & Section Offsets...")
    offsets_ms = [0]
    for i in range(len(segments) - 1):
        offsets_ms.append(offsets_ms[-1] + len(segments[i]) + 800)

    total_duration = offsets_ms[-1] + len(segments[-1]) + 1000
    master = AudioSegment.silent(duration=total_duration)
    for seg, off in zip(segments, offsets_ms):
        master = master.overlay(seg, position=off)

    master.export(str(MASTER_WAV), format="wav")
    print(f"  Master WAV ready: {MASTER_WAV} ({len(master)/1000.0:.1f}s)")

    print("Step 3: Extracting Millisecond-Exact Sentenced Subtitles...")
    srt_blocks = []
    cue_count = 1

    for text, base_offset_ms in zip(texts, offsets_ms):
        comm = edge_tts.Communicate(text, VOICE)
        async for chunk in comm.stream():
            if chunk["type"] == "SentenceBoundary":
                start_ms = base_offset_ms + (chunk["offset"] / 10000.0)
                dur_ms = chunk["duration"] / 10000.0
                end_ms = start_ms + dur_ms

                content = chunk["text"].strip()
                if content:
                    s_str = format_srt_timestamp(start_ms)
                    e_str = format_srt_timestamp(end_ms)
                    srt_blocks.append(f"{cue_count}\n{s_str} --> {e_str}\n{content}")
                    cue_count += 1

    srt_content = "\n\n".join(srt_blocks)
    PERFECT_SRT.write_text(srt_content, encoding="utf-8")
    
    # Save VTT format as well
    vtt_content = "WEBVTT\n\n" + srt_content.replace(",", ".")
    PERFECT_VTT.write_text(vtt_content, encoding="utf-8")
    print(f"  Subtitles saved: {PERFECT_SRT.name} ({cue_count-1} cues)")


def render_manim_and_burn_subtitles():
    print("Step 4: Rendering Manim Animation Scene...")
    manim_cmd = [
        sys.executable, "-m", "manim",
        "-ql", "-f",
        str(VIS_DIR / "vjepa_ch567_full.py"),
        "VJEPACh567Scene"
    ]
    subprocess.run(manim_cmd, check=True, cwd=str(ROOT_DIR))

    rendered_mp4 = MEDIA_DIR / "videos" / "vjepa_ch567_full" / "480p15" / "VJEPACh567Scene.mp4"
    if not rendered_mp4.exists():
        raise FileNotFoundError(f"Rendered video not found at {rendered_mp4}")

    print("Step 5: Hardcoding Subtitles onto MP4 Video via FFmpeg...")
    # Format path for FFmpeg subtitles filter on Windows
    srt_filter_path = str(PERFECT_SRT).replace("\\", "/").replace(":", "\\:")
    
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(rendered_mp4),
        "-vf", f"subtitles='{srt_filter_path}':force_style='FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,BorderStyle=4,MarginV=20'",
        "-c:a", "copy",
        str(FINAL_OUTPUT_MP4)
    ]
    subprocess.run(ffmpeg_cmd, check=True, cwd=str(ROOT_DIR))

    print("\n========================================================")
    print(f"🎉 SUCCESS! Subtitled video ready at:")
    print(f"👉 {FINAL_OUTPUT_MP4}")
    print("========================================================")


if __name__ == "__main__":
    asyncio.run(generate_tts_and_subtitles())
    render_manim_and_burn_subtitles()
