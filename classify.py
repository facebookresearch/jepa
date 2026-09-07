import json
import torch
from src.models.attentive_pooler import AttentiveClassifier

CLASSIFIER_CHECKPOINT_PATH: str = "/home/ubuntu/Desktop/lehack/models/vjepa/ssv2-probe.pth.tar"
CLASS_TO_INDEX_PATH: str = "/home/ubuntu/Desktop/lehack/datasets/something_something_v2/class_to_index.json"
VJEPA_LATENT: str = "/home/ubuntu/Desktop/lehack/v_jepa_embedding_inversion_attack/embeddings/train/VJEPA/2870.pt"


def main() -> None:
    classifier = load_classifier(CLASSIFIER_CHECKPOINT_PATH)
    index_to_class = load_index_to_class(CLASS_TO_INDEX_PATH)

    latent = torch.load(
        VJEPA_LATENT, 
        map_location="cuda"
    ).float()
    
    with torch.no_grad():
        logits = classifier.forward(latent)
        predicted = logits.argmax(dim=1)

    print(index_to_class[predicted.item()])


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
