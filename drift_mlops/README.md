# Real-Time Data Drift Detection & Mitigation Pipeline

Bilgisayar Mühendisliği Bitirme Projesi

---

## Bilgisayarına Nasıl Kurarsın?

### Adım 1: Python Kontrolü

Terminali (komut satırını) aç ve Python'un kurulu olduğunu kontrol et:

```bash
python --version
```

Python 3.9 veya üstü lazım. Eğer yoksa https://www.python.org/downloads/ adresinden indir.

Windows kullanıyorsan "Add Python to PATH" kutusunu işaretle.


### Adım 2: Proje Klasörünü İndir

Bu projeyi indirdiğinde `drift_mlops` adında bir klasör olacak.
Onu istediğin yere koy (mesela Masaüstü).

Terminalde o klasöre git:

```bash
# Windows
cd C:\Users\SENIN_ADIN\Desktop\drift_mlops

# Mac / Linux
cd ~/Desktop/drift_mlops
```


### Adım 3: Sanal Ortam Oluştur (Önerilir)

Bu adım zorunlu değil ama yapman iyi olur.
Sanal ortam = projenin kütüphanelerini sistemden ayırır, karışıklık olmaz.

```bash
# Sanal ortam oluştur
python -m venv venv

# Aktif et:
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

Aktif olduğunu anlarsın çünkü terminal satırının başında `(venv)` yazar.


### Adım 4: Gerekli Kütüphaneleri Kur

```bash
pip install -r requirements.txt
```

Bu komut şunları kurar:
- numpy (matematik işlemleri)
- pandas (tablo/veri işleme)
- scikit-learn (makine öğrenmesi modeli)
- scipy (istatistiksel testler)
- joblib (model kaydetme)

Hepsi toplam ~100 MB civarı, birkaç dakika sürer.


### Adım 5: Projeyi Çalıştır

```bash
# Klasörün BİR ÜST dizinine çık
cd ..

# Ana demo'yu çalıştır
python -m drift_mlops.main
```

ÖNEMLİ: `cd ..` ile bir üst klasöre çıkman lazım. Çünkü Python
`drift_mlops` klasörünü bir paket olarak görmesi gerekiyor.

Yani klasör yapın şöyle olmalı:
```
Desktop/
  └── drift_mlops/        ← python -m drift_mlops.main komutunu
        ├── main.py           BURANIN BİR ÜSTÜNDEn çalıştır
        ├── config/
        ├── data/
        ├── detection/
        ├── mitigation/
        ├── models/
        ├── pipeline/
        ├── experiments.py
        └── requirements.txt
```

Alternatif olarak doğrudan da çalıştırabilirsin:
```bash
cd drift_mlops
python main.py
```


### Adım 6: Deneyleri Çalıştır

```bash
python -m drift_mlops.experiments
```

Bu komut ~1-2 dakika sürer ve `experiment_results/` klasörüne
3 tane CSV dosyası kaydeder:
- algorithm_comparison.csv  (algoritma karşılaştırma tablosu)
- algorithm_ranking.csv     (genel sıralama)
- model_impact.csv          (drift'in model performansına etkisi)


---

## Proje Yapısı (Ne Nedir?)

```
drift_mlops/
│
├── config/
│   └── settings.py          # Tüm ayarlar ve eşik değerler burada
│                               Drift ne kadar olunca "tehlikeli" sayılacak?
│                               Model kaç ağaç kullansın? vs.
│
├── data/
│   ├── generator.py          # Sahte banka verisi üretir
│   │                           10 sütun: gelir, yaş, kredi skoru...
│   │                           Drift enjekte edebilir (ani, kademeli, tekrarlayan)
│   │
│   └── feature_store.py      # Referans veriyi saklar
│                               "Normalde veri nasıl görünüyordu?" sorusunun cevabı
│
├── detection/
│   ├── statistical.py        # 6 istatistiksel test
│   │                           KS, PSI, KL, JS, Chi-Square, Wasserstein
│   │                           Her biri "bu iki dağılım aynı mı?" sorusunu sorar
│   │
│   ├── streaming.py          # 3 gerçek zamanlı dedektör
│   │                           ADWIN, Page-Hinkley, DDM
│   │                           Veriyi tek tek işler, anında alarm verir
│   │
│   └── scorer.py             # Tüm test sonuçlarını birleştirir
│                               Tek bir skor verir: 0.0 (sorun yok) → 1.0 (ciddi drift)
│                               Seviye belirler: none / low / medium / high / critical
│
├── mitigation/
│   └── engine.py             # Drift tespit edilince ne yapılacağına karar verir
│                               low → sadece logla
│                               medium → uyar + modeli güncelle
│                               high → modeli sıfırdan eğit
│                               critical → yedek modele geç
│
├── models/
│   └── model_manager.py      # ML modelini eğitir, test eder, kaydeder
│                               Random Forest kullanır
│                               F1, accuracy, AUC gibi metrikleri hesaplar
│
├── pipeline/
│   └── orchestrator.py       # HER ŞEYİ BİRLEŞTİREN ANA DOSYA
│                               Veri gelir → drift kontrol → skor → müdahale → log
│
├── experiments.py            # Deneysel değerlendirme scripti
│                               12 senaryo × 9 algoritma karşılaştırması
│                               Rapor için hazır tablolar üretir
│
├── main.py                   # Demo: çalıştırınca tüm pipeline'ı gösterir
└── requirements.txt          # Gerekli kütüphaneler listesi
```


---

## Sık Karşılaşılan Sorunlar

### "ModuleNotFoundError: No module named 'drift_mlops'"
→ Yanlış klasördesin. `drift_mlops` klasörünün BİR ÜST dizininde olman lazım.
   Veya doğrudan `cd drift_mlops && python main.py` dene.

### "pip: command not found"
→ `pip3` dene. Veya `python -m pip install -r requirements.txt`

### Windows'ta "venv\Scripts\activate" çalışmıyor
→ PowerShell'de: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   yazıp tekrar dene.

### Çıktıda Türkçe karakterler bozuk görünüyor
→ Terminal encoding'ini UTF-8 yap:
   Windows: `chcp 65001`
