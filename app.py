from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labels import get_label
from src.modeling import ModelSpec, build_model
from src.utils import get_device, load_checkpoint


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@st.cache_resource
def load_model(ckpt_path: str):
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model", "resnet18")
    image_size = int(ckpt.get("image_size", 64))
    num_classes = int(ckpt["num_classes"])

    device = get_device()
    model = build_model(num_classes=num_classes, spec=ModelSpec(model_name))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, model_name, image_size, device


def predict(image: Image.Image, ckpt_path: str):
    model, model_name, image_size, device = load_model(ckpt_path)
    tfm = make_transform(image_size)
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_probs, top_indices = torch.topk(probs, k=min(5, probs.numel()))
    top_rows = [
        {
            "Sinif ID": int(idx),
            "Etiket": get_label(int(idx)),
            "Guven": float(prob),
        }
        for idx, prob in zip(top_indices, top_probs)
    ]
    return model_name, top_rows


def main() -> None:
    st.set_page_config(page_title="Trafik Isareti Siniflandirma", page_icon="🚦", layout="wide")

    st.title("Trafik İşareti Sınıflandırma Demo")
    st.write(
        "Bir trafik işareti görseli yükleyin. Model görseli sınıflandırıp en olası sınıfları gösterecek."
    )

    ckpt_path = st.sidebar.text_input("Model checkpoint", value="runs/latest.pt")
    st.sidebar.markdown("### Sunum Akışı")
    st.sidebar.write("1. Görsel yükle")
    st.sidebar.write("2. Model tahminini göster")
    st.sidebar.write("3. Güven oranını ve top-5 sınıfı açıkla")

    if not Path(ckpt_path).exists():
        st.error(
            "Model dosyası bulunamadı. Önce eğitim komutunu çalıştırın: "
            "`python scripts/train.py --dataset gtsrb --epochs 1 --batch-size 64`"
        )
        return

    uploaded = st.file_uploader("Trafik işareti fotoğrafı yükle", type=["png", "jpg", "jpeg", "ppm"])

    if uploaded is None:
        st.info("Demo için bir trafik işareti fotoğrafı yükleyin.")
        st.code(
            'python scripts/predict.py --ckpt runs/latest.pt --image "path/to/image.ppm"',
            language="powershell",
        )
        return

    image = Image.open(uploaded).convert("RGB")
    model_name, top_rows = predict(image, ckpt_path)
    best = top_rows[0]

    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.subheader("Yüklenen Görsel")
        st.image(image, use_container_width=True)

    with col_result:
        st.subheader("Model Tahmini")
        st.metric("Tahmin Edilen Sınıf", f"{best['Sinif ID']} - {best['Etiket']}")
        st.metric("Güven Oranı", f"{best['Guven'] * 100:.2f}%")
        st.caption(f"Kullanılan model: {model_name}")

    st.subheader("En Olası 5 Sınıf")
    df = pd.DataFrame(top_rows)
    df["Guven (%)"] = df["Guven"] * 100
    st.dataframe(df[["Sinif ID", "Etiket", "Guven (%)"]], use_container_width=True)
    st.bar_chart(df.set_index("Etiket")["Guven (%)"])

    st.success("Tahmin tamamlandı. Bu ekran sunumda canlı demo olarak gösterilebilir.")


if __name__ == "__main__":
    main()

