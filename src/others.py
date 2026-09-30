import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from get_features import get_features
from get_features import get_features_mexican


def main():

    data = scipy.io.loadmat('data.mat')
    audio_train, audio_test = data['audio_train'].T, data['audio_test'].T
    labels_train, labels_test = data['labels_train'][:,0], data['labels_test'][:,0]
    fs = data['fs'][0,0]

    # EXTRAGERE FEATURES
    feat_gabor_train = get_features(audio_train, fs)
    feat_gabor_test  = get_features(audio_test, fs)

    feat_mex_train = get_features_mexican(audio_train, fs)
    feat_mex_test  = get_features_mexican(audio_test, fs)

    results = {}

    # ---------------- KNN + Gabor ----------------
    knn = KNeighborsClassifier()
    knn.fit(feat_gabor_train, labels_train)
    acc = np.mean(knn.predict(feat_gabor_test) == labels_test)
    results["KNN + Gabor"] = acc

    # ---------------- KNN + Mexican Hat ----------------
    knn = KNeighborsClassifier()
    knn.fit(feat_mex_train, labels_train)
    acc = np.mean(knn.predict(feat_mex_test) == labels_test)
    results["KNN + Mexican Hat"] = acc

    # ---------------- SVM + Gabor ----------------
    svm = SVC()
    svm.fit(feat_gabor_train, labels_train)
    acc = np.mean(svm.predict(feat_gabor_test) == labels_test)
    results["SVM + Gabor"] = acc

    # ---------------- SVM + Mexican Hat ----------------
    svm = SVC()
    svm.fit(feat_mex_train, labels_train)
    acc = np.mean(svm.predict(feat_mex_test) == labels_test)
    results["SVM + Mexican Hat"] = acc

    # ---------------- AFISARE ----------------
    for k, v in results.items():
        print(f"{k}: {v:.2f}")

    # ---------------- GRAFIC ----------------
    import os

    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy")
    plt.title("Comparatie clasificatori si filtre")
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs("img", exist_ok=True)

    plt.savefig("img/comparatie_clasificatori.png")
    plt.close()


if __name__ == "__main__":
    main()
