# BÀI TẬP LỚN
# MÔN HỌC: TRÍ TUỆ NHÂN TẠO VÀ HỌC MÁY
# ĐỀ TÀI : NHẬN DẠNG CỬ CHỈ TAY ĐIỀU KHIỂN TRÒ RẮN SĂN MỒI
## HỌ VÀ TÊN SINH VIÊN: LƯƠNG VĂN HỌC
## LỚP: K58KTP
## MSSV: K225480106025
## GIÁO VIÊN HƯỚNG DẪN: TS. NGUYỄN TUẤN LINH


Dự án game rắn điều khiển bằng cử chỉ tay (MediaPipe + Machine Learning).

## 1) Yêu cầu môi trường

- Python 3.10+ (khuyến nghị 3.11)
- Webcam hoạt động tốt
- Hệ điều hành Windows (có thể chạy trên OS khác nếu cài đủ thư viện)

## 2) Cài đặt thư viện
Mở terminal tại thư mục dự án.

Thư viện cần cài đã được liệt kê trong file `requirements.txt`.

### Cách nhanh

```bash
pip install -r requirements.txt
```

### Cách khuyến nghị (dùng virtual environment)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Cấu trúc file chính

- `thu_thap_du_lieu.py`: Thu thập dữ liệu cử chỉ tay và lưu `.npy`
- `huan_luyen_svm.py`: Huấn luyện mô hình SVM
- `huan_luyen_knn.py`: Huấn luyện mô hình KNN (nếu cần)
- `danh_gia_mo_hinh.py`: Đánh giá SVM/KNN, in metric và vẽ biểu đồ
- `dieu_khien_ran.py`: Chạy game rắn điều khiển bằng cử chỉ

## 4) Hướng dẫn chạy code

### Bước 1: Thu thập dữ liệu

```bash
python thu_thap_du_lieu.py
```

Nhập nhãn khi được hỏi (`len`, `xuong`, `trai`, `phai`), làm lại cho từng nhãn.
Nhấn `ESC` để dừng mỗi lần thu thập.

### Bước 2: Huấn luyện mô hình

Huấn luyện SVM:

```bash
python huan_luyen_svm.py
```

Nếu bạn muốn thử KNN:

```bash
python huan_luyen_knn.py
```

### Bước 3: Đánh giá mô hình

```bash
python danh_gia_mo_hinh.py
```

Script sẽ:
- Chia train/test công bằng
- Train SVM và KNN trên cùng tập train
- In Accuracy + Classification Report
- Vẽ confusion matrix và biểu đồ so sánh

### Bước 4: Chạy game rắn

```bash
python dieu_khien_ran.py
```

Cử chỉ tay để điều hướng:
- `len` -> đi lên
- `xuong` -> đi xuống
- `trai` -> đi trái
- `phai` -> đi phải

<img width="587" height="315" alt="image" src="https://github.com/user-attachments/assets/68e2a3d9-f992-4684-ba0e-046e43e9a15a" />

<img width="587" height="315" alt="image" src="https://github.com/user-attachments/assets/e9a6992b-ec6f-4f18-aa73-9802ec56c760" />

Giao diện thu thập dữ liệu cử chỉ tay

## 5) Giao diện trò chơi

Giao diện game được thiết kế theo phong cách tối hiện đại để dễ quan sát khi chơi bằng cử chỉ tay:

- Nền gradient có lưới nhẹ giúp theo dõi vị trí rắn trên bàn chơi.
- Rắn có màu đầu/thân khác nhau, bo góc để nhìn rõ hướng di chuyển.
- HUD hiển thị các chỉ số quan trọng: `Score`, `Gesture`, `FPS`, `Frame ms`, `Speed`.
- Màn hình `Game Over` có lớp mờ và nút `Chơi lại` rõ ràng.

<img width="1605" height="810" alt="image" src="https://github.com/user-attachments/assets/6a7f44bf-812a-4272-9e06-f3f8de48c287" />

Màn hình trò chơi

<img width="1573" height="773" alt="image" src="https://github.com/user-attachments/assets/a417e1f2-48e5-4763-96ec-f93aeec550a8" />

Màn hình game over

## 6) Lỗi thường gặp

- Lỗi mở webcam:
  - Đóng app đang dùng camera (Zoom, Teams, trình duyệt...)
- Không tìm thấy model:
  - Chạy lại `python huan_luyen_svm.py` để tạo file model trong `mo_hinh/`
- Dữ liệu quá ít:
  - Thu thêm mẫu cho mỗi nhãn để model ổn định hơn

## 7) Quy trình khuyến nghị

1. Thu dữ liệu cho đủ 4 nhãn
2. Huấn luyện model
3. Đánh giá model
4. Chạy game và tinh chỉnh tiếp
