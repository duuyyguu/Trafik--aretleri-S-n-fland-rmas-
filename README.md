# Trafik İşaretleri Sınıflandırması (Bilgisayar Görmesi Ödevi)

Bu repo, trafik işareti görüntülerini sınıflandıran bir modeli eğitmek ve değerlendirmek için hazırlanmıştır.

## 3 Kişilik Ekip Görev Dağılımı

### Duygu — Proje sahibi / ML pipeline (Sprint-1)
- Repo iskeleti, çalışma akışı, standartlar
- Dataset indirme + dataloader
- Baseline model (transfer learning) eğitimi
- Train/validation/test loader uyumu
- Tekrarlanabilir validation split (`--seed`, `--val-split`)
- Eğitim geçmişi ve grafik çıktıları (`history.json`, accuracy/loss curves)
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
- `train.py` ile eğitim ve validation takibi, `eval.py` ile test metrikleri
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

Eğitim sonunda `runs/` altında bir deney klasörü oluşur:
- `best.pt`: validation accuracy'ye göre en iyi model
- `last.pt`: son epoch checkpoint'i
- `history.json`: epoch bazlı train/validation metrikleri
- `run.json`: eğitim ayarları

### Eğitim Grafiklerini Üretme
```bash
python scripts/plot_history.py --history runs/<run-klasoru>/history.json
```

Bu komut aynı run klasörüne `accuracy_curve.png` ve `loss_curve.png` kaydeder.

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

### Görsel Web Demo (Sunum İçin)
```bash
streamlit run app.py
```

Bu komut tarayıcıda basit bir arayüz açar. Fotoğraf yükleyerek modelin tahmin ettiği sınıfı, güven oranını ve en olası 5 sınıfı görsel olarak gösterebilirsiniz.

## Repo Yapısı
- `scripts/`: CLI script’leri (train/eval/predict)
- `src/`: ortak kod (data, model, utils)
- `data/`: dataset klasörü (git’e dahil edilmez)
- `runs/`: checkpoint ve loglar (git’e dahil edilmez)

## Duygu'nun Teslim Ettiği Kısım

Duygu'nun sorumluluğu, modelin uçtan uca çalışmasını sağlayan ML pipeline bölümüdür. Bu kapsamda:
- `src/data.py`: GTSRB dataset loader, train/validation/test ayrımı ve tekrarlanabilir split
- `scripts/train.py`: transfer learning eğitimi, validation takibi, early stopping, scheduler, checkpoint kaydı
- `scripts/eval.py`: test seti değerlendirmesi, `metrics.json` ve `confusion_matrix.png` üretimi
- `scripts/predict.py`: tek görsel üzerinde sınıf tahmini
- `scripts/plot_history.py`: eğitim süreci için accuracy/loss grafikleri
- `app.py`: sunum için fotoğraf yüklemeli web demo arayüzü

Okan rapor kısmında `runs/` çıktılarındaki `history.json`, `metrics.json`, `accuracy_curve.png`, `loss_curve.png` ve `confusion_matrix.png` dosyalarını kullanabilir.

