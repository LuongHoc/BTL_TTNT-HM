import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ================== CAU HINH CHUNG ==================
DATA_DIR = "du_lieu"
NHAN_LIST = ["len", "xuong", "trai", "phai"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
KNN_NEIGHBORS = 5


def tai_du_lieu() -> Tuple[np.ndarray, np.ndarray]:
    # Khoi tao danh sach chua dac trung (X) va nhan (y)
    X = []
    y = []

    # Duyet du lieu theo tung nhan de dam bao nhan/duong dan ro rang
    for nhan in NHAN_LIST:
        thu_muc = os.path.join(DATA_DIR, nhan)

        # Bo qua neu thu muc nhan khong ton tai
        if not os.path.isdir(thu_muc):
            continue

        # Chi doc cac file .npy de tranh tep rac
        for ten_file in sorted(os.listdir(thu_muc)):
            if not ten_file.endswith(".npy"):
                continue

            du_lieu = np.load(os.path.join(thu_muc, ten_file))
            X.append(du_lieu)
            y.append(nhan)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Khong tim thay du lieu de danh gia trong thu muc du_lieu/")

    return X, y


def tao_mo_hinh() -> Dict[str, object]:
    # Khoi tao cac mo hinh can so sanh
    return {
        "SVM": SVC(kernel="linear"),
        "KNN": KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    }


def tinh_chi_so(y_true, y_pred) -> Dict[str, float]:
    # Tinh cac chi so co ban de danh gia mo hinh
    return {
        "accuracy": accuracy_score(y_true, y_pred)
    }


def ve_confusion_matrix(cm: np.ndarray, tieu_de: str, labels, figure_index: int) -> None:
    # Ve confusion matrix cho mot mo hinh
    plt.figure(figure_index)
    plt.imshow(cm, cmap="Blues")
    plt.title(tieu_de)
    plt.colorbar()

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Du doan")
    plt.ylabel("Thuc te")


def in_bao_cao_chi_tiet(ten_mo_hinh: str, y_test, y_pred_test, acc_train: float, acc_test: float) -> None:
    # In ket qua tong quan va report chi tiet cho tung mo hinh
    print(f"\n--- {ten_mo_hinh} ---")
    print(f"Train accuracy: {acc_train:.4f}")
    print(f"Test accuracy:  {acc_test:.4f}")
    print(f"\nClassification report - {ten_mo_hinh}")
    print(classification_report(y_test, y_pred_test, labels=NHAN_LIST, zero_division=0))


def main() -> None:
    # 1) Tai du lieu va chia train/test
    X, y = tai_du_lieu()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 2) Tao cac mo hinh can so sanh
    cac_mo_hinh = tao_mo_hinh()

    # 3) Train + du doan + tinh chi so cho tung mo hinh
    ket_qua = {}
    for ten, mo_hinh in cac_mo_hinh.items():
        mo_hinh.fit(X_train, y_train)

        y_pred_train = mo_hinh.predict(X_train)
        y_pred_test = mo_hinh.predict(X_test)

        ket_qua[ten] = {
            "y_pred_test": y_pred_test,
            "train_metrics": tinh_chi_so(y_train, y_pred_train),
            "test_metrics": tinh_chi_so(y_test, y_pred_test)
        }

    # 4) In thong tin tong quan tap du lieu
    print("=== KET QUA DANH GIA CONG BANG (train/test split) ===")
    print(f"Tong so mau: {len(X)}")
    print(f"So mau train: {len(X_train)} | So mau test: {len(X_test)}")

    # 5) In bao cao chi tiet cho tung mo hinh
    for ten in ["SVM", "KNN"]:
        in_bao_cao_chi_tiet(
            ten,
            y_test,
            ket_qua[ten]["y_pred_test"],
            ket_qua[ten]["train_metrics"]["accuracy"],
            ket_qua[ten]["test_metrics"]["accuracy"]
        )

    # 6) Ve confusion matrix cho tung mo hinh
    for index, ten in enumerate(["SVM", "KNN"], start=1):
        cm = confusion_matrix(y_test, ket_qua[ten]["y_pred_test"], labels=NHAN_LIST)
        ve_confusion_matrix(cm, f"Confusion Matrix - {ten}", NHAN_LIST, figure_index=index)

    # 7) Ve bieu do cot so sanh test accuracy
    plt.figure(3)
    models = ["SVM", "KNN"]
    accuracy = [
        ket_qua["SVM"]["test_metrics"]["accuracy"],
        ket_qua["KNN"]["test_metrics"]["accuracy"]
    ]

    plt.bar(models, accuracy, color=["#4f81bd", "#f39c3d"])

    for i, acc in enumerate(accuracy):
        plt.text(i, acc + 0.01, f"{acc:.3f}", ha="center")

    plt.title("So sanh do chinh xac (Test)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)

    plt.show()


if __name__ == "__main__":
    # Khoi chay chuong trinh va thong bao loi than thien neu co su co
    try:
        main()
    except Exception as loi:
        print(f"Loi khi danh gia mo hinh: {loi}")