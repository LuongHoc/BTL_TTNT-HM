import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.svm import SVC


# ===== Cau hinh duong dan va nhan =====
DATA_DIR = "du_lieu"
MODEL_PATH = "mo_hinh/svm_tay.pkl"
CAC_NHAN = ["len", "xuong", "trai", "phai"]


def tai_du_lieu() -> Tuple[np.ndarray, np.ndarray]:
    # Khoi tao danh sach chua dac trung (X) va nhan (y)
    X = []
    y = []

    # Duyet tung thu muc nhan de lay toan bo file .npy
    for nhan in CAC_NHAN:
        duong_dan = os.path.join(DATA_DIR, nhan)

        if not os.path.isdir(duong_dan):
            continue

        for tep in sorted(os.listdir(duong_dan)):
            if not tep.endswith(".npy"):
                continue

            du_lieu = np.load(os.path.join(duong_dan, tep))

            X.append(du_lieu)
            y.append(nhan)

    return np.array(X), np.array(y)


def huan_luyen() -> None:
    # 1) Tai du lieu huan luyen
    X, y = tai_du_lieu()

    if len(X) == 0:
        raise ValueError("Khong tim thay du lieu huan luyen trong thu muc du_lieu/")

    # 2) Tao va huan luyen mo hinh SVM
    mo_hinh = SVC(kernel="linear")
    mo_hinh.fit(X, y)

    # 3) Luu model da huan luyen de dung cho game
    joblib.dump(mo_hinh, MODEL_PATH)

    # 4) In thong tin tong ket
    print(f"Da huan luyen xong. So mau: {len(X)}")
    print(f"Model da luu tai: {MODEL_PATH}")


def main() -> None:
    # Diem vao chuong trinh: bat dau huan luyen
    huan_luyen()


if __name__ == "__main__":
    # Chay huan luyen va thong bao loi than thien neu co
    try:
        main()
    except Exception as loi:
        print(f"Loi khi huan luyen mo hinh: {loi}")