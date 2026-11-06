# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603484
- **Họ và tên:** Nguyễn Hải Anh
- **SEED:** 6200

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

Bảng số liệu thực nghiệm đo đạc trực tiếp từ quá trình huấn luyện và kiểm thử trên tập test (775 tin, 92 spam) và tập cân bằng SUB (40 tin):

| Phương án | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Thông lượng (tin/s) | p50 (ms/tin) | p95 (ms/tin) | USD / 1k tin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. BoW + LogReg (tự code)** | 0.986 | 0.761 | 0.859 | 0.947 | 2.35s | 69,972 | 0.030 ms | 0.055 ms | $0.00 |
| **2. BoW + MLP (deep learning)** | 0.975 | 0.848 | 0.907 | 0.947 | 2.46s | 68,957 | 0.113 ms | 0.133 ms | $0.00 |
| **3a. Encoder (MiniLM) + kNN** | 0.833 | 0.924 | 0.876 | 0.974 | 50.89s | 39 | 39.837 ms | 76.024 ms | $0.00 |
| **3b. Encoder (MiniLM) + LogReg** | 0.966 | 0.913 | 0.939 | 0.974 | 51.38s | 46 | 21.237 ms | 28.290 ms | $0.00 |
| **3c. Encoder (MiniLM) + MLP head**| 0.966 | 0.924 | 0.944 | 0.974 | 51.57s | 46 | 20.837 ms | 28.968 ms | $0.00 |

*Ghi chú:*
- Hàng 3b và 3c tái sử dụng thuật toán tối ưu của Phần 1 và Phần 2 nhưng chạy trên vector embedding 384 chiều của MiniLM.
- Thời gian train của 3a/3b/3c bao gồm ~50.9s để encoder mã hóa toàn bộ 3.619 tin nhắn tập train.
- Độ trễ p50/p95 đo từ chuỗi ký tự văn bản thô cho đến nhãn đầu ra, mô phỏng đúng một request sản phẩm thực tế.

---

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất và cơ chế giải thích:

1. **Dự đoán `hard_eval` (Stress test 50 case khó):**
   - *Tôi đoán:* F1 của các mô hình trên 50 case khó chỉ giảm nhẹ xuống khoảng 0.50 – 0.70.
   - *Thực tế:* Mô hình BoW + LogReg tụt thảm hại xuống **F1 = 0.333** (Accuracy 52%), BoW + MLP chỉ đạt **F1 = 0.421**.
   - *Nguyên nhân lệch:* Tôi đã đánh giá quá cao khả năng nhận diện của từ khóa BoW. Khi gặp các case `natural_spam` (viết lịch sự như rủ đi chơi, thông báo việc làm) hoặc `obfuscation` (cố tình viết 'C0ngr4ts', 'F R E E'), mô hình BoW hoàn toàn "mù" vì các từ này không hề xuất hiện trong 2.000 từ của tập train.

2. **Dự đoán `ml_f1` (Mô hình Bag-of-Words + Logistic Regression):**
   - *Tôi đoán:* F1 đạt khoảng 0.88 – 0.92, Recall và Precision tương đương nhau.
   - *Thực tế:* F1 thực tế chỉ đạt **0.859**; đặc biệt có sự lệch pha cực lớn giữa **Precision rất cao (0.986)** nhưng **Recall thấp (0.761)**.
   - *Nguyên nhân lệch:* Các từ khóa spam mạnh (như 'claim', 'prize', 'call', 'txt') kéo trọng số dương rất lớn nên hễ mô hình chặn tin nào là tin đó hầu như chắc chắn là spam (Precision ~99%). Tuy nhiên, nhiều tin spam sử dụng từ ngữ đời thường hoặc câu từ ngắn gọn không chứa từ khóa top đầu, khiến mô hình bỏ sót gần 24% lượng tin rác (Recall thấp).

3. **Dự đoán `dl_relu` (Bỏ kích hoạt ReLU trong mạng MLP):**
   - *Tôi đoán:* Bỏ ReLU thì F1 sẽ sụt giảm sâu do mất khả năng phi tuyến.
   - *Thực tế:* F1 khi không có ReLU vẫn đạt xấp xỉ ~0.86 – 0.88, tương đương với Logistic Regression 1 lớp.
   - *Nguyên nhân lệch:* Về mặt toán học, hai lớp Linear liên tiếp $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$ bản chất suy biến thành đúng một phép biến đổi affine tuyến tính $W'x + b'$, tức trở lại bài toán Logistic Regression ban đầu chứ không làm sập mô hình.

---

## 4. Phân tích 50 case khó

- **Model tụt điểm nhiều nhất:** 
  - Mô hình **1. BoW + LogReg** tụt điểm nặng nề nhất: từ **F1 = 0.859** (test thường) rơi tự do xuống **F1 = 0.333** (giảm 61.2% hiệu năng).
  - Nhóm bị tụt nhiều nhất là **`natural_spam`** (accuracy chỉ đạt **10%**) và **`obfuscation`** (accuracy chỉ đạt **17%**).
- **Nguyên nhân tập test thường "quá lạc quan":** 
  Do tập test được lấy cùng phân phối với tập train và dữ liệu SMS gốc chứa nhiều tin nhắn spam theo mẫu (template) chỉ khác số điện thoại liên hệ hoặc mã dự thưởng. Mô hình chỉ cần "học vẹt" vài từ khóa trùng lặp là đã đạt điểm cao. Khi gặp 50 case viết tay độc lập với bẫy ngữ nghĩa, bản chất mô hình mới lộ rõ.

- **Hai case cụ thể khiến mô hình sai và cơ chế:**
  1. **Case #12 (`natural_spam` - Thật là SPAM):**
     > *"hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for some fun, text me back if you're free"*
     - *Cơ chế sai:* Mở đầu cực kỳ lịch sự, ngôn từ tự nhiên của một cuộc trò chuyện cá nhân ('hi there', 'sorry to bother you', 'new in town'). Không hề có bất kỳ từ khóa spam truyền thống nào ('win', 'urgent', 'prize'). Cả BoW và MLP đều phân loại nhầm thành **HAM** với xác suất spam gần bằng 0.
  2. **Case #1 (`ham_looks_spam` - Thật là HAM):**
     > *"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"*
     - *Cơ chế sai:* Tin nhắn chứa hàng loạt từ khóa spam điển hình có trọng số dương rất lớn trong mô hình BoW: `'congrats'`, `'won'`, `'call'`. Do đó BoW cộng dồn điểm số và kiên quyết phân loại thành **SPAM**, mặc dù ngữ cảnh thực tế chỉ là hai người bạn cá cược một chầu bia (`'owe you a beer'`). Trong khi đó, các mô hình Embedding (3b, 3c) hiểu được ngữ cảnh bạn bè đùa vui nên phân loại chính xác là HAM.

---

## 5. Thử thách cá nhân

- **Mã đề nhận được (SEED = 6200):** **Đề #3**
  > *"DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?"*

- **Cách thực hiện:**
  Khởi tạo hai mô hình MLP với kích thước tầng ẩn `hidden=4` và `hidden=512`, giữ nguyên số epoch (15) và learning rate (1e-3) trên cùng tập dữ liệu BoW 2.000 chiều:
  - Mạng nhỏ: `build_mlp(2000, hidden=4)`
  - Mạng lớn: `build_mlp(2000, hidden=512)`

- **Kết quả thực nghiệm:**
  - **Mạng `hidden = 4`:**
    - Số tham số: $2000 \times 4 + 4 + 4 \times 1 + 1 = \mathbf{8.009}$ tham số.
    - Thời gian huấn luyện: **0.88 giây**.
    - Hiệu năng: **F1 test = 0.912** (Precision = 0.975, Recall = 0.859).
  - **Mạng `hidden = 512`:**
    - Số tham số: $2000 \times 512 + 512 + 512 \times 1 + 1 = \mathbf{1.025.025}$ tham số (> 1 triệu tham số).
    - Thời gian huấn luyện: **8.37 giây** (chậm gấp gần 10 lần).
    - Hiệu năng: **F1 test = 0.912** (không tăng một phần nghìn nào so với mạng 4 nơ-ron).

- **Giải thích & Kết luận:**
  1. Đầu vào là Bag-of-Words với 2.000 từ rời rạc. Bản thân tín hiệu phân loại spam đã rất tuyến tính; việc tăng số nơ-ron ẩn từ 4 lên 512 làm số tham số phình to hơn 128 lần nhưng không mang lại biểu diễn ngữ nghĩa mới, ngược lại còn làm tăng nguy cơ overfit nặng và tốn tài nguyên tính toán.
  2. **Khi nào nên chọn mạng nhỏ:**
     - Triển khai trên các thiết bị biên (Edge AI, smartphone, gateway viễn thông) có giới hạn khắt khe về RAM và chip xử lý.
     - Ứng dụng yêu cầu độ trễ cực thấp (Ultra-low latency, real-time inference < 1ms).
     - Khi tập dữ liệu huấn luyện còn khiêm tốn, mạng nhỏ giúp kiểm soát chặt chẽ dung lượng mô hình, tránh học thuộc lòng dữ liệu nhiễu.

---

## 6. Phương án cho sản phẩm thật

*Yêu cầu bài toán:* Nhà mạng viễn thông cần lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình), yêu cầu nghiêm ngặt **p95 < 50 ms/tin**.

### 1. Phân tích và lựa chọn phương án:
- **Kiểm tra tiêu chí độ trễ SLA (p95 < 50 ms):**
  - `1. BoW + LogReg`: p95 = 0.055 ms (Đạt SLA, cực nhanh).
  - `2. BoW + MLP`: p95 = 0.133 ms (Đạt SLA, rất nhanh).
  - `3a. Encoder + kNN`: p95 = 76.024 ms (**Vi phạm SLA**, loại bỏ).
  - `3b. Encoder + LogReg`: p95 = **28.290 ms** (Đạt SLA, F1 test = 0.939).
  - `3c. Encoder + MLP head`: p95 = **28.968 ms** (Đạt SLA, F1 test = 0.944).
  - `LLM (gọi qua API)`: Độ trễ trung bình > 500 – 1500 ms/request (**Vi phạm nghiêm trọng SLA**, chi phí khổng lồ).

- **Phương án lựa chọn:** Chọn **`3b. Encoder (MiniLM) + LogReg`** (hoặc `3c. Encoder + MLP head`).
  - *Lý do:* Vừa thỏa mãn điều kiện độ trễ khắt khe ($p95 \approx 28.3\text{ ms} < 50\text{ ms}$), vừa đạt chất lượng phân loại cao nhất ($F1 = 0.939 - 0.944$).
  - *Chấp nhận đánh đổi:* Cần cấp phát bộ nhớ RAM (~90MB cho mô hình MiniLM) trên server và chi phí tính toán GPU/CPU cao hơn BoW, nhưng hoàn toàn xứng đáng vì khả năng bắt spam ngụy trang vượt trội.

### 2. Ước tính chi phí nếu dùng LLM:
- Giả sử dùng model rẻ nhất là `gpt-6-luna` ($0.10 input / $0.50 output trên 1M token), mỗi SMS trung bình ~100 token in và 1 token out:
  - Chi phí 1 ngày (10 triệu SMS): $(10 \times 100 \times 0.10) + (10 \times 1 \times 0.50) = 100 + 5 = \mathbf{105\text{ USD/ngày}}$.
  - Chi phí 1 tháng (30 ngày): $105 \times 30 = \mathbf{3.150\text{ USD/tháng}}$ (~**80.000.000 VNĐ/tháng**).
  - Nếu dùng model lớn (`gpt-6.1-terra` với giá gấp 10 lần): chi phí lên tới $\mathbf{31.500\text{ USD/tháng}}$ (~**800.000.000 VNĐ/tháng**), hoàn toàn không khả thi về mặt kinh tế.

### 3. Thiết kế hệ thống định tuyến lai (Router Cascade Architecture):
Nhằm kết hợp ưu điểm của các thế hệ, đề xuất kiến trúc lọc 2 tầng:
```text
Tin nhắn SMS đến 
   ├──► Tầng 1: Bộ lọc nhanh MiniLM + LogReg (độ trễ ~28ms, $0)
   │       ├── Xác suất p >= 0.85  ──► Chặn ngay (SPAM chắc chắn)  [~15% traffic]
   │       ├── Xác suất p <= 0.15  ──► Cho qua ngay (HAM chắc chắn) [~80% traffic]
   │       └── Xác suất 0.15 < p < 0.85 (Vùng lưỡng lự) [~5% traffic]
   │               │
   │               └──► Tầng 2: LLM Agent với Prompt phòng thủ (xử lý chuyên sâu)
```
- **Lợi ích vượt trội:** Xử lý được 95% lưu lượng tin nhắn trong < 30ms với chi phí 0 USD; chỉ 5% tin nhắn phức tạp mới gọi LLM, giúp tiết kiệm **95% chi phí API** (chỉ còn ~$150/tháng) trong khi vẫn đảm bảo độ chính xác tối đa ở các ca lừa đảo tinh vi.

---

## 7. AI log và điều đã học

### AI Log (Khai báo minh bạch các lần dùng AI hỗ trợ):
1. **Lần 1 — Khái niệm & Toán học Gradient Descent (Phần 1):**
   - *Hỏi:* Cơ chế toán học của vector gradient trong hàm loss Binary Cross-Entropy và vì sao bước cập nhật lại lấy $X^T(p - y) / n$?
   - *Nhận:* AI giải thích đạo hàm từng phần của hàm mất mát log-loss theo $w$, chỉ ra ý nghĩa trực giác của $(p - y)$ là sai số dự đoán.
   - *Tự làm & chỉnh sửa:* Tự viết vòng lặp huấn luyện bằng NumPy thuần mà không dùng bất kỳ thư viện hỗ trợ ngoài nào; tự kiểm tra dấu của bước cập nhật trọng số.
2. **Lần 2 — Khắc phục bẫy Reasoning Model trong LLM API (Phần 4):**
   - *Hỏi:* Vì sao khi gọi mô hình reasoning mới với `max_tokens=3` lại bị trả về chuỗi rỗng?
   - *Nhận:* AI giải thích cơ chế "thinking tokens" ngầm của các reasoning model (GPT-6, Gemini 3, Groq) nuốt hết quota token nếu không tắt suy luận.
   - *Tự làm & chỉnh sửa:* Thêm tham số `reasoning_effort="none"` và chuyển sang `max_completion_tokens` trong hàm `call_openai` và viết hàm `parse_label` xử lý an toàn chuỗi nhãn.
3. **Lần 3 — Phân tích lỗi trên 50 case khó (Phần 5):**
   - *Hỏi:* Các kỹ thuật tấn công Prompt Injection trong tin nhắn SMS và cơ chế phòng thủ hiệu quả?
   - *Nhận:* AI gợi ý kỹ thuật đóng gói dữ liệu đầu vào trong thẻ phân cách `<sms>...</sms>` để tách biệt chỉ thị hệ thống và dữ liệu người dùng.
   - *Tự làm & chỉnh sửa:* Áp dụng prompt phòng thủ vào hàm `build_defended_prompt` và tự phân tích bảng lỗi theo từng nhóm bẫy ngữ nghĩa.

### Điều hiểu ra hôm nay:
1. **Biểu diễn dữ liệu (Representation) quan trọng hơn độ phức tạp của bộ phân loại:** Khi chuyển từ Bag-of-Words sang Pre-trained Embedding, chất lượng mô hình tăng vọt từ F1 = 0.859 lên 0.939 mà chỉ cần một bộ phân loại tuyến tính đơn giản (Logistic Regression). Thêm nhiều tầng ẩn (MLP) trên vector embedding hầu như không cải thiện thêm nhiều.
2. **Đừng bao giờ tin tưởng mù quáng vào Accuracy và tập test cùng phân phối:** Accuracy 98% có thể hoàn toàn vô nghĩa trên tập dữ liệu lệch lớp; và điểm F1 0.94 trên tập test chuẩn lập tức sụp đổ xuống 0.33 khi đối mặt với các tin nhắn viết tay có bẫy ngữ nghĩa thực tế.
3. **Sản phẩm AI thực tế là bài toán đánh đổi đa chiều:** Không có mô hình nào "toàn năng"; một kỹ sư AI giỏi là người biết cân bằng giữa F1-score, độ trễ p95, thông lượng tính toán, chi phí tài chính và độ bền vững trước các đòn tấn công đối kháng.
