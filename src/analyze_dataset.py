from __future__ import annotations
from pathlib import Path
from torchvision import datasets
import matplotlib.pyplot as plt
import collections

GTSRB_CLASSES = {
    0: "Hız Limiti 20", 1: "Hız Limiti 30", 2: "Hız Limiti 50",
    3: "Hız Limiti 60", 4: "Hız Limiti 70", 5: "Hız Limiti 80",
    6: "Hız Limiti 80 Sonu", 7: "Hız Limiti 100", 8: "Hız Limiti 120",
    9: "Geçiş Yasak", 10: "Kamyon Geçiş Yasak", 11: "Öncelikli Yol",
    12: "Öncelik Ver", 13: "Dur", 14: "Taşıt Giremez",
    15: "Kamyon Giremez", 16: "Giriş Yasak", 17: "Genel Tehlike",
    18: "Sol Viraj", 19: "Sağ Viraj", 20: "Çift Viraj",
    21: "Engebeli Yol", 22: "Kaygan Yol", 23: "Dar Yol Sağ",
    24: "Yol Çalışması", 25: "Trafik Işıkları", 26: "Yaya Geçidi",
    27: "Okul Geçidi", 28: "Bisiklet Geçidi", 29: "Buz/Kar",
    30: "Yaban Hayvanı", 31: "Hız Limiti Sonu", 32: "Sağa Dön",
    33: "Sola Dön", 34: "Düz Git", 35: "Düz veya Sağa",
    36: "Düz veya Sola", 37: "Sağdan Geç", 38: "Soldan Geç",
    39: "Dönel Kavşak", 40: "Geçiş Yasağı Sonu", 41: "Kamyon Yasağı Sonu",
    42: "Tüm Yasaklar Sonu",
}

def analyze(data_dir: str = "data") -> None:
    root = Path(data_dir) / "gtsrb"
    print("Dataset indiriliyor / kontrol ediliyor...")
    ds = datasets.GTSRB(root=str(root), split="train", download=True)

    # Sınıf sayılarını hesapla
    labels = [lbl for _, lbl in ds._samples]
    counter = collections.Counter(labels)

    print(f"\nToplam görüntü: {len(labels)}")
    print(f"Toplam sınıf: {len(counter)}")
    print(f"\nEn az örnekli 5 sınıf:")
    for cls_id, count in counter.most_common()[-5:]:
        print(f"  Sınıf {cls_id} ({GTSRB_CLASSES[cls_id]}): {count} örnek")
    print(f"\nEn çok örnekli 5 sınıf:")
    for cls_id, count in counter.most_common(5):
        print(f"  Sınıf {cls_id} ({GTSRB_CLASSES[cls_id]}): {count} örnek")

    # Grafik
    ids = sorted(counter.keys())
    counts = [counter[i] for i in ids]

    plt.figure(figsize=(16, 5))
    plt.bar(ids, counts, color="steelblue")
    plt.xlabel("Sınıf ID")
    plt.ylabel("Görüntü Sayısı")
    plt.title("GTSRB Sınıf Dağılımı")
    plt.tight_layout()
    plt.savefig("assets/class_distribution.png", dpi=150)
    print("\nGrafik kaydedildi: assets/class_distribution.png")
    print("\nGrafik kaydedildi: runs/class_distribution.png")

if __name__ == "__main__":
    analyze()