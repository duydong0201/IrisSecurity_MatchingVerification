# IrisSecurity - Matching & Verification

Hệ thống sinh mã mống mắt (IrisCode) từ ảnh, huấn luyện ngưỡng (threshold) tối ưu và đánh giá/xác thực kết quả.

### 1) Yêu cầu hệ thống
- Python 3.9+ (khuyến nghị 3.10/3.11)
- pip
- Windows PowerShell (hoặc terminal bất kỳ)

### 2) Cài đặt môi trường
Khởi tạo môi trường ảo và cài dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install opencv-python numpy scikit-learn
```
### 3) Cấu trúc dữ liệu đầu vào
Đặt ảnh vào thư mục `data` theo cấu trúc:
data/
<person_id>/
L/.jpg # ảnh mắt trái
R/.jpg # ảnh mắt phải
í dụ: `data/person1/L/xxx.jpg`, `data/person1/R/yyy.jpg`

- Ảnh đầu vào: `.jpg`
- Mỗi ảnh sẽ được chuyển thành một file `.npy` (IrisCode) trong `codes_cropped/`.

### 4) Sinh IrisCode (.npy) từ ảnh
Chạy pipeline tách mống mắt + Log-Gabor + nhị phân hóa để sinh mã:
```powershell
python gen_cropped_npy.py --data_dir data --out_dir codes_cropped
```
- Kết quả: rất nhiều file `.npy` trong `codes_cropped/`, tên dạng: `<person>_<L|R>_<tên_ảnh>.npy`

Lưu ý: Thư mục `codes_cropped/` có thể rất lớn.

### 5) Huấn luyện ngưỡng tối ưu (threshold)
Chạy huấn luyện nhiều lần (nhiều epoch) để tìm ngưỡng tối ưu theo Macro-F1:
```powershell
python train_threshold.py
```
- Tạo file:
  - `best_threshold.txt`  (chứa ngưỡng trung bình)
  - `test_list.txt`       (danh sách mẫu test đã dùng)

Mặc định script dùng dữ liệu trong `codes_cropped/`. Có thể chỉnh trong code nếu cần.

### 6) Đánh giá nhanh theo ngưỡng đã học
Dùng `test_eval.py` để tính Accuracy/Precision/Recall/F1 với `best_threshold.txt`:
```powershell
python test_eval.py
```
- Script sẽ đọc `best_threshold.txt`, `test_list.txt` và in các chỉ số.

### 7) Thử xác thực ngẫu nhiên một mẫu
`verify_eval.py` sẽ:
- Lấy ngẫu nhiên 1 file từ `test_list.txt`
- Xây template của đúng người từ tập train
- Tính Hamming distance và so với `best_threshold.txt`

Chạy:
```powershell
python verify_eval.py
```

### 8) Xác thực trực tiếp từ ảnh PNG/JPG lẻ
Nếu bạn có 1 ảnh lẻ và muốn kiểm tra có khớp 1 `person_id` nào đó (dựa trên `codes_cropped/`):
```powershell
python -c "import iris_seg_eval as m; ok, dist = m.verify_from_image('path/to/image.png', 'person1', threshold=0.2); print('MATCH' if ok else 'NOT MATCH', 'dist=', dist)"
```
- `person1` là tiền tố tên file `.npy` trong `codes_cropped/` (ví dụ `person1_L_...npy`)
- `threshold` có thể thay bằng giá trị trong `best_threshold.txt` (khuyến nghị dùng kết quả huấn luyện).
  - Ví dụ:
    ```powershell
    $th = Get-Content best_threshold.txt
    python -c "import iris_seg_eval as m; th=float('$th'); ok, dist=m.verify_from_image('path/to/image.png','person1',threshold=th); print(ok, dist)"
    ```

### 9) Ghi chú kỹ thuật
- Các hàm chính:
  - Tách mống mắt và lọc: `iris_seg_eval.segment_iris`, `iris_seg_eval.log_gabor_filter`
  - Sinh mã nhị phân: `iris_seg_eval.iris_to_code`
  - So khớp: Hamming distance
- Template của mỗi người được tạo bằng trung vị/giá trị điển hình từ tập train.
- Ngưỡng được tối ưu theo F1/Macro-F1 tùy script.

### 10) Lỗi thường gặp
- Không có dữ liệu: đảm bảo `data/<person>/<L|R>/*.jpg` tồn tại trước khi chạy bước 4.
- Không tìm thấy `.npy` cho `person_id` khi xác thực ảnh lẻ: cần sinh đủ `.npy` trong `codes_cropped/` (bước 4).
- OpenCV không cài được: cập nhật `pip`, cài lại `opencv-python`.

### 11) Lệnh nhanh (tổng hợp)
```powershell
# 1) Môi trường
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # hoặc: pip install opencv-python numpy scikit-learn

# 2) Sinh .npy
python gen_cropped_npy.py --data_dir data --out_dir codes_cropped

# 3) Huấn luyện threshold
python train_threshold.py

# 4) Đánh giá
python test_eval.py

# 5) Thử xác thực một mẫu bất kỳ
python verify_eval.py
```
```

- Bạn có thể điều chỉnh `num_persons`, `test_size`, hoặc cách chọn template trong các script nếu cần tối ưu thêm.
- Dự án đã có sẵn `codes_cropped/` và `data/` (nếu repository của bạn chứa), chạy trực tiếp từ bước 5.
- Nếu dataset lớn, các bước sinh `.npy` và huấn luyện có thể mất nhiều thời gian.
- Bạn có thể điều chỉnh num_persons, test_size, hoặc cách chọn template trong các script nếu cần tối ưu thêm.
- Dự án đã có sẵn codes_cropped/ và data/ (nếu repository của bạn chứa), chạy trực tiếp từ bước 5.
- Nếu dataset lớn, các bước sinh .npy và huấn luyện có thể mất nhiều thời gian.

- - -
