import numpy as np
from get_features import get_features
from sklearn.neighbors import KNeighborsClassifier
import scipy


def main():
    # load data from matfile 'data.mat'
    data = scipy.io.loadmat('data.mat')
    audio_train, audio_test = data['audio_train'].T,  data['audio_test'].T
    labels_train, labels_test = data['labels_train'], data['labels_test']
    fs = data['fs'][0,0]


    # setul de data este impartit in 2 parti: train si test
    # vom calcula setul de trasaturi pentru ambele seturi
    # apoi vom folosi o metoda de machine learning antrenata pe setul de train si 
    # vom evalua performanta folosind setul de test


    # Pentru a calcula rezultatele mai rapid, putem folosi doar o fractiune din
    # fiecare semnal audio. Rezulatele vor fi mai proaste, dar timpul de calcul va
    # fi mai mic. 

    alpha = 1.0 # 0.1
    start1 = audio_train.shape[1] // 2 - int(alpha * audio_train.shape[1] // 2) + 1
    end1 = audio_train.shape[1] // 2 + int(alpha * audio_train.shape[1] // 2)
    audio_train_small = audio_train[:, start1:end1]

    start2 = audio_test.shape[1] // 2 - int(alpha * audio_test.shape[1] // 2) + 1    
    end2 = audio_test.shape[1] // 2 + int(alpha * audio_test.shape[1] // 2)    
    audio_test_small = audio_test[:, start2:end2]   

    # Dimensiunile datelor ar trebui sa fie:
    # audio_train_small: [D1, N]
    # audio_test_small: [D2, N]
    # labels_train: [D1, 1]
    # labels_test: [D2, 1]


    # calculam vectorii de trasaturi pentru fiecare fisier din datasetul de train si de  test
    # functia `get_features`` primeste toate sunetele dintr-un set de date intr-o matrice
    # de dimensiune Dimensiune_dataset (D) x Numaresults_esantioane (N) [D x N] si returneaza toate
    # trasaturile (features) acestor sunete intr-o matrice de dimensiune  [D x (2*M)]

    # filters: [M, F]
    # feat_train: [D1, 2M]
    # feat_test: [D2, 2M]
    # TODO: calculati trasaturile
    feat_train = get_features(audio_train_small, fs)
    feat_test = get_features(audio_test_small, fs)

    labels_train = labels_train[:,0]
    labels_test = labels_test[:,0]

    clf = KNeighborsClassifier()
    clf.fit(feat_train, labels_train)
    pred_train = clf.predict(feat_train)
    pred_test = clf.predict(feat_test)
    acc_train = np.mean(pred_train == labels_train)
    acc_test = np.mean(pred_test == labels_test)
    print(f'Accuracy on train: {acc_train:.2f}')
    print(f'Accuracy on test:  {acc_test:.2f}')

# main function
if __name__ == "__main__":
    main()