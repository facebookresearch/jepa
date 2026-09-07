import json
from pathlib import Path

import torch

from src.models.attentive_pooler import AttentiveClassifier

# Classifier head
CLASSIFIER_CHECKPOINT_PATH: str = "/home/ubuntu/Desktop/lehack/models/vjepa/ssv2-probe.pth.tar"
CLASS_TO_INDEX_PATH: str = "/home/ubuntu/Desktop/lehack/datasets/something_something_v2/class_to_index.json"

ORIGINAL_LATENTS_DIRECTORY: str = "/home/ubuntu/Desktop/lehack/v_jepa_embedding_inversion_attack/embeddings/train/VJEPA"
PREDICTED_VIDEOS_DIRECTORY: str = "/home/ubuntu/Desktop/lehack/v_jepa_embedding_inversion_attack/videos/predicted"

ORIGINAL_VIDEOS_DIRECTORY: str = "/home/ubuntu/Desktop/lehack/datasets/something_something_v2/videos"

OUTPUT_PATH: str = "/home/ubuntu/Desktop/lehack/LTX-2/pairs.jsonl"


def main() -> None:
    classifier = load_classifier(CLASSIFIER_CHECKPOINT_PATH)
    index_to_class = load_index_to_class(CLASS_TO_INDEX_PATH)

    pairs: list[dict[str, str]] = []
    for path in Path(PREDICTED_VIDEOS_DIRECTORY).iterdir():
        p: dict[str, str] = {}

        # Video to end with (produce)
        p["video"] = ORIGINAL_VIDEOS_DIRECTORY + f"/{path.name}"
        # Video to start with (denoise)
        p["reference_video"] = PREDICTED_VIDEOS_DIRECTORY + f"/{path.name}"

        predicted_latent = torch.load(
            ORIGINAL_LATENTS_DIRECTORY + f"/{path.stem}.pt", 
            map_location="cuda"
        )
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = classifier(predicted_latent)
                predicted_caption = logits.argmax(dim=1)

        # Caption acts as text guide
        p["caption"] = index_to_class[predicted_caption.item()]

        pairs.append(p)

    with open(OUTPUT_PATH, "w") as outfile:
        for item in pairs:
            outfile.write(json.dumps(item) + "\n")

    print(f"Generated {len(pairs)} pairs at {OUTPUT_PATH}")



def load_classifier(ckpt_path: str) -> AttentiveClassifier:
    classifier = AttentiveClassifier(
        embed_dim=1024, # ViT-L
        num_heads=16, # ViT-L
        depth=1,
        num_classes=174 # SSv2
    ).to("cuda")

    classifier_ckpt = torch.load(ckpt_path)

    state_dict = classifier_ckpt["classifier"]
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }
    classifier.load_state_dict(state_dict)

    return classifier.eval()


def load_index_to_class(path: str) -> dict[int, str]:
    with open(path, "r") as infile:
        class_to_index = json.load(infile)
    index_to_class = {
        int(v): k for k, v in class_to_index.items()
    }
    return index_to_class


if __name__ == "__main__":
    main()