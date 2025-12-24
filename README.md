# Domain Generation Algorithm Detection: A Comparative Analysis of LSTM, Transformer, and Feature-Based Approaches

## 📋 Mô Tả Dự Án

Dự án này xây dựng các mô hình **học máy** để phát hiện và phân loại các miền tên được tạo bằng thuật toán DGA (Domain Generation Algorithm). DGA là kỹ thuật được sử dụng bởi malware để tạo ra các tên miền động nhằm tránh bị chặn bởi các danh sách đen truyền thống.

### Mục Tiêu
- **Phát hiện DGA**: Xác định liệu một tên miền có phải được tạo bằng DGA hay không
- **So sánh Mô Hình**: Đánh giá hiệu suất của các mô hình khác nhau (Random Forest, LSTM, Transformer)
- **Mô Phỏng Tấn Công**: Sinh ra các miền tên DGA thực tế từ 30+ thuật toán DGA khác nhau

---

## 🏗️ Cấu Trúc Thư Mục

```
dga_predict/
├── run.py                          # Script chính: chạy các thí nghiệm
├── regenerate_data.py              # Tái tạo dataset với DGA dựa trên ML
├── test_dga_generators.py          # Kiểm tra các DGA generators
├── verify_real_dgas.py             # Xác nhận hiệu suất các DGA thực
├── LICENSE                         # Giấy phép GPL v2
│
├── dga_classifier/
│   ├── __init__.py
│   ├── data.py                     # Tạo dataset (benign + DGA domains)
│   ├── manual_rf.py                # Mô hình Random Forest
│   ├── lstm.py                     # Mô hình LSTM (Deep Learning)
│   ├── transformer.py              # Mô hình Transformer
│   ├── bigram.py                   # Phân tích bigram
│   │
│   └── dga_generators/             # 30+ DGA generators
│       ├── real_lstm_dga.py        # DGA dựa trên LSTM được huấn luyện
│       ├── real_gan_dga.py         # DGA dựa trên GAN
│       ├── adversarial_trained_dga.py  # DGA huấn luyện đối kháng
│       ├── hash_dga.py             # DGA dựa trên hash
│       ├── context_dga.py          # DGA có ngữ cảnh
│       ├── multistage_dga.py       # DGA nhiều giai đoạn
│       ├── neural_like_dga.py      # DGA mô phỏng mạng nơ-ron
│       ├── banjori.py, kraken.py, lockyv2.py  # DGA malware thực
│       └── [26 DGA khác...]
```

---

## 🚀 Cách Sử Dụng

### 1. Chuẩn Bị Dữ Liệu

Để tái tạo dataset với các DGA dựa trên ML được huấn luyện đúng cách:

```bash
python regenerate_data.py
```

**Các bước thực hiện:**
- Tải xuống 1M miền tên từ Alexa Top Domains (hoặc tạo benign domains tổng hợp)
- Huấn luyện các mô hình DGA thực (LSTM, GAN, Adversarial)
- Sinh ra ~1K miền DGA từ các mô hình đó
- Tạo tệp `traindata.pkl` chứa dữ liệu huấn luyện

⏱️ **Thời gian:** 10-30 phút (tùy vào cấu hình hệ thống)

### 2. Chạy Các Thí Nghiệm

```bash
python run.py
```

**Tùy chọn:**
```bash
# Chạy tất cả mô hình
python run.py

# Chạy riêng Random Forest
python run.py --manualrf --no-lstm --no-transformer

# Chạy riêng LSTM
python run.py --no-manualrf --lstm --no-transformer

# Chạy riêng Transformer
python run.py --no-manualrf --no-lstm --transformer

# Thay đổi số lần cross-validation (mặc định: 10)
python run.py --nfolds 5

# Buộc tái tạo dữ liệu
python run.py --force
```

**Kết quả:**
- `results.pkl` - Kết quả được lưu trong bộ nhớ đệm
- `metrics.csv` - Bảng chi tiết các chỉ số
- `roc_*.png` - Đồ thị ROC curve
- `confusion_*.png` - Ma trận nhầm lẫn

### 3. Kiểm Tra DGA Generators

```bash
python test_dga_generators.py
```

Kiểm tra xem tất cả các DGA generators có hoạt động đúng hay không.

### 4. Xác Nhận Hiệu Suất DGA Thực

```bash
python verify_real_dgas.py
```

Kiểm tra các mô hình DGA dựa trên ML (LSTM, GAN, Adversarial).

---

## 📊 Các Mô Hình Được Hỗ Trợ

### 1. **Random Forest (Manual RF)**
- **Loại:** Machine Learning cổ điển
- **Tính năng:** Độ dài miền, tần suất ký tự, entropy, bigram
- **Ưu điểm:** Nhanh, dễ giải thích
- **Nhược điểm:** Không học được các mẫu phức tạp

### 2. **LSTM (Long Short-Term Memory)**
- **Loại:** Deep Learning
- **Tính năng:** Học chuỗi ký tự trực tiếp (character-level)
- **Ưu điểm:** Bắt được các mối quan hệ dài hạn
- **Nhược điểm:** Chậm hơn, cần GPU để huấn luyện nhanh

### 3. **Transformer**
- **Loại:** Deep Learning (Attention-based)
- **Tính năng:** Attention mechanism để học các mối quan hệ song song
- **Ưu điểm:** Tính toán nhanh, hiệu suất cao
- **Nhược điểm:** Cần nhiều dữ liệu hơn

---

## 🦠 Các Thuật Toán DGA Được Hỗ Trợ

### **DGA Dựa Trên Hash**
- `hash_dga.py` - Tiêu chuẩn DGA dựa trên hash
- `advanced_hash_dga.py` - Hash nâng cao với nhiều quy tắc

### **DGA Tinh Xảo**
- `context_dga.py` - DGA có ngữ cảnh
- `multistage_dga.py` - DGA nhiều giai đoạn
- `time_varying_dga.py` - DGA thay đổi theo thời gian
- `obfuscated_dga.py` - DGA có che phủ

### **DGA Dựa Trên Neural Network**
- `real_lstm_dga.py` - LSTM thực được huấn luyện
- `real_gan_dga.py` - GAN tạo sinh miền tên DGA
- `adversarial_trained_dga.py` - DGA đối kháng

### **DGA Malware Thực**
- `banjori.py` - Banjori malware
- `kraken.py` - Kraken botnet
- `lockyv2.py` - Locky v2 ransomware
- `cryptolocker.py` - CryptoLocker ransomware
- `qakbot.py` - QakBot malware
- `pykspa.py` - Pykspa botnet
- [Và 20+ thuật toán DGA khác từ malware thực]

---

## 📈 Các Chỉ Số Đánh Giá

Dự án sử dụng các chỉ số sau để đánh giá mô hình:

| Chỉ Số | Ý Nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ dự đoán đúng |
| **Precision** | Trong các DGA dự đoán, bao nhiêu đúng là DGA |
| **Recall** | Trong tất cả các DGA thực, bao nhiêu được phát hiện |
| **F1-Score** | Trung bình điều hòa của Precision và Recall |
| **ROC-AUC** | Diện tích dưới đường cong ROC |
| **FPR** | Tỷ lệ dương tính giả |
| **TPR** | Tỷ lệ dương tính đúng |

---

## 🔧 Yêu Cầu Hệ Thống

### Python Packages
```
numpy>=1.19.0
scikit-learn>=0.23.0
tensorflow>=2.0.0  # Hoặc torch nếu dùng PyTorch
matplotlib>=3.3.0
tldextract>=2.2.0
```

### Cài Đặt
```bash
pip install -r requirements.txt
```

### Tài Nguyên
- **RAM:** Tối thiểu 8GB (16GB khuyên dùng)
- **Disk:** ~2GB cho dataset
- **CPU:** 4+ cores
- **GPU:** Tùy chọn (để tăng tốc độ huấn luyện)

---

## 📊 Ví Dụ Kết Quả

```
╔════════════════════════════════════════════════════════════════╗
║              DGA Classification Results                         ║
╚════════════════════════════════════════════════════════════════╝

Manual Random Forest:
  Accuracy:  95.2%
  Precision: 94.8%
  Recall:    95.6%
  F1-Score:  95.2%
  ROC-AUC:   0.985

LSTM Neural Network:
  Accuracy:  97.3%
  Precision: 96.9%
  Recall:    97.8%
  F1-Score:  97.3%
  ROC-AUC:   0.995

Transformer:
  Accuracy:  96.8%
  Precision: 96.4%
  Recall:    97.3%
  F1-Score:  96.8%
  ROC-AUC:   0.992
```

---

## 🔬 Chi Tiết Kỹ Thuật

### Xử Lý Dữ Liệu

**Dataset chuẩn:**
- **Benign domains:** 1,000,000 miền từ Alexa Top Domains
- **DGA domains:** ~1,000 miền từ 30+ thuật toán DGA khác nhau
- **Tỷ lệ:** ~1:1000 (cân bằng)

**Các tính năng (Features):**

#### Random Forest:
1. Độ dài miền
2. Entropy Shannon
3. Tần suất từng ký tự (a-z, 0-9)
4. Bigram và trigram frequency
5. Số lượng vowel/consonant

#### LSTM/Transformer:
1. Character embedding (30 chiều)
2. Chuỗi ký tự đầu vào: độ dài 32-128

### Huấn Luyện

```python
# Chia dữ liệu
- Training: 80%
- Testing: 20%
- Cross-validation: 10-fold

# Siêu tham số
- Random Forest: 100 trees, max_depth=50
- LSTM: 2 layers, 128 units, dropout=0.2
- Transformer: 4 heads, 256 hidden, 2 layers
```

---

## 🚨 Vấn Đề Thường Gặp

### 1. Lỗi "Không tải được Alexa domains"
**Giải pháp:** Dự án sẽ tự động tạo benign domains tổng hợp

### 2. Bộ nhớ không đủ
**Giải pháp:** Giảm số lượng domains:
```python
# Trong data.py, thay đổi:
NUM_BENIGN = 500000  # Thay vì 1000000
```

### 3. LSTM/Transformer chậm
**Giải pháp:** 
- Dùng GPU: `CUDA_VISIBLE_DEVICES=0 python run.py`
- Giảm batch size
- Dùng Random Forest để kiểm tra nhanh

---

## 📚 Tài Liệu Tham Khảo

- [Domain Generation Algorithms (DGA)](https://en.wikipedia.org/wiki/Domain_generation_algorithm)
- [Random Forest in Machine Learning](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [LSTM Networks](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Transformer Models](https://huggingface.co/docs/transformers/)
- [Endgame - DGA Detection Dataset](https://github.com/endgameinc/domain_generation_algorithms)

---

## 📝 Giấy Phép

Dự án này được phát hành dưới giấy phép **GNU General Public License v2.0** (GPLv2).

Xem [LICENSE](LICENSE) để biết chi tiết.

---

## ✍️ Tác Giả & Đóng Góp

Nếu bạn có đề xuất cải thiện hoặc phát hiện lỗi, vui lòng mở issue hoặc pull request.

---

## 🎯 Các Cải Tiến Tương Lai

- [ ] Hỗ trợ GPU acceleration cho tất cả mô hình
- [ ] Triển khai mô hình XGBoost
- [ ] API REST cho phát hiện DGA real-time
- [ ] Dashboard web để trực quan hóa kết quả
- [ ] Hỗ trợ các DGA generator mới
- [ ] Fine-tuning các mô hình pre-trained

---

**Cập nhật lần cuối:** Tháng 12, 2025
