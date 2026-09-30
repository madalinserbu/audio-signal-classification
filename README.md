# README - Audio signal classification

---

## 1. Scopul proiectului

Scopul acestui proiect este analiza si clasificarea semnalelor audio folosind filtre de tip Gabor si un filtru alternativ ales (Mexican Hat), urmate de extragerea trasaturilor statistice si aplicarea unor clasificatori de tip machine learning. Proiectul urmareste evidentierea influentei tipului de filtru si a clasificatorului asupra performantei sistemului.

---

## 2. Structura proiectului

```
src/
│
├ tema_2025_schelet.py       # fisierul principal primit (nemodificat)
├ others.py                  # the other experiments
├ get_features.py            # extragerea trasaturilor
├ gabor_filter.py            # implementare filtre Gabor
├ mexican_hat.py             # filtru alternativ
├ plot_filters.py            # generare grafice filtre
└ data.mat                   # setul de date
img/
├ *_gabor_cos.png
├ *_gabor_sin.png
├ *_spectru_filtre.png
├ *_spectru_mexican.png
└ comparatie_clasificatori.png
```

---

## 3. Implementarea filtrelor

### 3.1 Filtru Gabor

Filtrul Gabor este implementat conform ecuatiei din enunt, ca produs intre o functie Gaussiana si o functie cos / sin cu frecventa centrala.

Fisier: `gabor_filter.py`

* functia `gabor_filter()` creeaza un singur filtru Gabor
* functia `create_gabor_bank()` genereaza o banca de M filtre distribuite pe scala Mel
* filtrele sunt normalizate pentru stabilitate numerica

Scala Mel este folosita pentru a simula perceptia auditiva umana, astfel incat distributia frecventelor este mai densa in zona joasa.

---

### 3.2 Filtru alternativ - Mexican Hat

Filtrul Mexican Hat (Ricker wavelet) este utilizat ca alternativa la Gabor.

Fisier: `mexican_hat.py`
Caracteristici:

* este derivata a doua a unei Gaussiene
* comportament de tip band-pass
* componenta DC este eliminata
* filtrul este normalizat

Acest filtru este utilizat atat pentru analiza vizuala, cat si pentru extragerea trasaturilor in experimentele extinse.

---

## 4. Afisarea filtrelor si a spectrelor

Fisier: `plot_filters.py`

Sunt generate automat urmatoarele grafice:

* forma in timp a filtrului Gabor cos
* forma in timp a filtrului Gabor sin
* spectrul tuturor filtrelor Gabor
* spectrul filtrului Mexican Hat

Spectrele sunt afisate folosind FFT si sunt salvate in folderul `img/` sub forma PNG.

---

## 5. Extragerea trasaturilor

Fisier: `get_features.py`

Semnalele sunt impartite in ferestre suprapuse, iar pe fiecare fereastra se aplica filtrarea.

Pentru fiecare filtru se calculeaza:

* media raspunsurilor
* deviatia standard

### 5.1 Gabor

Functia `get_features()` genereaza trasaturi de dimensiune 2M pentru fiecare semnal, unde:

* M = numar filtre Gabor
* 2M = mean + std

### 5.2 Mexican Hat

Functia `get_features_mexican()` foloseste un singur filtru Mexican Hat si produce 2 trasaturi:

* media
* deviatia standard

---

## 6. Clasificare

### 6.1 KNN + Gabor (schelet obligatoriu)

Fisier: `tema_2025_schelet.py`

Rezultat obtinut:

```
Accuracy on train: 0.70
Accuracy on test:  0.55
```

Acest rezultat se incadreaza in intervalul cerut de enunt.

---

### 6.2 Experimente extinse

Fisier: `others.py`

Au fost realizate urmatoarele combinatii:

| Clasificator | Filtru      | Accuracy |
| ------------ | ----------- | -------- |
| KNN          | Gabor       | 0.55     |
| KNN          | Mexican Hat | 0.34     |
| SVM          | Gabor       | 0.61     |
| SVM          | Mexican Hat | 0.29     |

Clasificatorul SVM impreuna cu filtrul Gabor a oferit cea mai buna performanta.

---

## 7. Grafic comparativ

A fost generat graficul `comparatie_clasificatori.png` care afiseaza performanta tuturor combinatiilor:

* KNN + Gabor
* KNN + Mexican Hat
* SVM + Gabor
* SVM + Mexican Hat

Graficul confirma superioritatea combinatiei SVM + Gabor.

---

## 8. Observatii si concluzii

* Filtrele Gabor sunt mai eficiente decat Mexican Hat pentru recunoasterea sunetelor eligibile.
* Scala Mel contribuie la o distributie mai relevanta a frecventelor.
* Clasificatorul SVM ofera o capacitate mai buna de separare fata de KNN.
* Combinatia optima obtinuta: SVM + Gabor.

---

Rezultatele obtinute demonstreaza importanta alegerii corecte a filtrului si a clasificatorului in procesarea semnalelor audio.

---
