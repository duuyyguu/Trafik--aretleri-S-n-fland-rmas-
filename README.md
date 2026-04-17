# Trafik İşaretleri Sınıflandırması (Bilgisayar Görmesi Ödevi)

Bu repo, trafik işareti görüntülerini sınıflandıran bir modeli eğitmek ve değerlendirmek için hazırlanmıştır.

## 3 Kişilik Ekip Görev Dağılımı

### Duygu — Proje sahibi / ML pipeline (Sprint-1)
- Repo iskeleti, çalışma akışı, standartlar
- Dataset indirme + dataloader
- Baseline model (transfer learning) eğitimi
- Değerlendirme metrikleri, confusion matrix çıktıları
- Tek komutla eğitim/değerlendirme script’leri

### Reyhan — Veri & Ön-işleme / Augmentation
- Hedef dataset seçimi ve sınıf etiket eşlemesi (GTSRB veya dersin verdiği dataset)
- Veri analizi (sınıf dağılımı, örnek görseller, dengesizlik)
- Augmentation stratejisi (color jitter, random affine, blur vb.)
- Eğitim için train/val/test split mantığı ve raporlama

### Okan — Rapor / Deney tasarımı & iyileştirme
- Deney planı (baseline vs iyileştirmeler)
- Hiperparametre araması (lr, batch size, optimizer, scheduler)
- Model karşılaştırması (ResNet18 vs MobileNetV3 gibi)
- Sonuçların raporlanması (tablolar, grafikler, hata analizi)

## Sprint-1 (Başlangıç) Hedefi
- GTSRB dataset ile çalışan bir baseline kurulum
- `train.py` ile eğitim, `eval.py` ile test metrikleri
- `predict.py` ile tek görsel tahmini

> Not: Dersiniz farklı bir dataset veriyorsa `src/data.py` içindeki dataset kısmını uyarlayacağız.

## Kurulum

Python 3.10+ önerilir.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Çalıştırma

### Eğitim
```bash
python scripts/train.py --dataset gtsrb --epochs 8 --batch-size 64 --scheduler cosine --patience 3
```

### Değerlendirme
```bash
python scripts/eval.py --dataset gtsrb --ckpt runs/latest.pt
```

Değerlendirme sonunda `runs/` altında bir klasöre şu çıktılar kaydedilir:
- `metrics.json`
- `confusion_matrix.png`

### Tahmin
```bash
python scripts/predict.py --ckpt runs/latest.pt --image "path/to/image.png"
```

## Repo Yapısı
- `scripts/`: CLI script’leri (train/eval/predict)
- `src/`: ortak kod (data, model, utils)
- `data/`: dataset klasörü (git’e dahil edilmez)
- `runs/`: checkpoint ve loglar (git’e dahil edilmez)

