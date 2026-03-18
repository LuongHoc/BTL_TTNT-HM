import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Optional, List

# ===== Cau hinh chung =====
DATA_DIR = "du_lieu"
VALID_LABELS = ["len", "xuong", "trai", "phai"]
WINDOW_NAME = "thu thap du lieu"
ESC_KEY = 27

# ===== Khoi tao Mediapipe =====
mp_tay = mp.solutions.hands
tay = mp_tay.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
ve = mp.solutions.drawing_utils

def lay_diem_tay(ket_qua) -> Optional[List[float]]:
    # Lay vector dac trung (x, y) tu 21 moc ban tay
    if not ket_qua.multi_hand_landmarks:
        return None

    for ban_tay in ket_qua.multi_hand_landmarks:
        diem = []

        for d in ban_tay.landmark:
            diem.append(d.x)
            diem.append(d.y)

        return diem

    return None


def thu_thap_du_lieu(nhan: str) -> None:
    # Chuan hoa nhan va kiem tra tinh hop le
    nhan = nhan.strip().lower()
    if nhan not in VALID_LABELS:
        raise ValueError(f"Nhan khong hop le: {nhan}. Chi nhan: {', '.join(VALID_LABELS)}")

    # Tao thu muc luu du lieu theo nhan neu chua ton tai
    thu_muc = os.path.join(DATA_DIR, nhan)
    os.makedirs(thu_muc, exist_ok=True)

    # Mo camera de thu thap du lieu theo thoi gian thuc
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Khong mo duoc camera")

    dem = len([f for f in os.listdir(thu_muc) if f.endswith(".npy")])
    print(f"Bat dau thu thap nhan '{nhan}'. Nhan ESC de dung.")

    while True:
        # 1) Doc khung hinh va tien xu ly
        ret, khung = camera.read()
        if not ret:
            break

        khung = cv2.flip(khung, 1)
        khung_rgb = cv2.cvtColor(khung, cv2.COLOR_BGR2RGB)

        # 2) Phat hien ban tay va trich xuat dac trung
        ket_qua = tay.process(khung_rgb)
        diem = lay_diem_tay(ket_qua)

        # 3) Luu du lieu vao file .npy neu phat hien tay
        if diem is not None:
            file_path = os.path.join(thu_muc, f"{dem}.npy")
            np.save(file_path, diem)
            dem += 1

        # 4) Ve moc tay va thong tin so mau de de theo doi
        if ket_qua.multi_hand_landmarks:
            for ban_tay in ket_qua.multi_hand_landmarks:
                ve.draw_landmarks(khung, ban_tay, mp_tay.HAND_CONNECTIONS)

        cv2.putText(
            khung,
            f"Nhan: {nhan} | So mau: {dem}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow(WINDOW_NAME, khung)

        # 5) Nhan ESC de ket thuc
        if cv2.waitKey(1) == ESC_KEY:
            break

    # 6) Giai phong tai nguyen camera/cua so
    camera.release()
    cv2.destroyAllWindows()
    print(f"Da luu tong cong {dem} mau cho nhan '{nhan}'.")


def main() -> None:
    # Diem vao chuong trinh: nhap nhan va bat dau thu thap
    nhan = input(f"Nhap nhan ({'/'.join(VALID_LABELS)}): ")

    thu_thap_du_lieu(nhan)


if __name__ == "__main__":
    # Chay thu thap du lieu va thong bao loi than thien neu co
    try:
        main()
    except Exception as loi:
        print(f"Loi khi thu thap du lieu: {loi}")