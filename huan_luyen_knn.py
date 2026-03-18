import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

X = []
y = []

nhan_list = ["len", "xuong", "trai", "phai"]

for nhan in nhan_list:
    thu_muc = os.path.join("du_lieu", nhan)
    for file in os.listdir(thu_muc):
        du_lieu = np.load(os.path.join(thu_muc, file))
        X.append(du_lieu)
        y.append(nhan)

X = np.array(X)
y = np.array(y)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Lưu model
joblib.dump(model, "mo_hinh/knn_tay.pkl")

print("Đã train và lưu model KNN")