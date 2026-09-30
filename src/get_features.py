import numpy as np
from gabor_filter import create_gabor_bank


# ================================
# Segmentarea semnalului in ferestre
# ================================
# Imparte semnalul 1D in ferestre suprapuse de lungime fixa.
# Acest pas simuleaza o analiza locala in timp (similar STFT).
#
# x         -> semnal audio
# frame_len -> lungimea ferestrei
# hop       -> pasul dintre ferestre (suprapunere controlata)
def framed_signal(x, frame_len, hop):

    N = len(x)

    # Daca semnalul este mai scurt decat fereastra, nu putem construi ferestre valide
    if N < frame_len:
        return np.zeros((0, frame_len))

    # Numarul de ferestre care pot fi generate
    F = 1 + (N - frame_len) // hop

    # Forma matricei de iesire: F ferestre x frame_len
    shape = (F, frame_len)

    # Strides folosit pentru eficienta (fara copiere de date)
    strides = (x.strides[0] * hop, x.strides[0])

    try:
        # metoda rapida folosind stride tricks
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()
    except:
        # fallback sigur daca metoda optimizata esueaza
        frames = np.zeros((F, frame_len))
        for i in range(F):
            frames[i, :] = x[i*hop : i*hop + frame_len]
        return frames


# ================================
# Extragerea caracteristicilor din banca de filtre
# ================================
# Aplica toate filtrele asupra tuturor ferestrelor
# si calculeaza statistici (mean si std) pentru fiecare filtru.
def compute_features_from_bank(frames, filters):

    # Daca nu exista ferestre, returnam vectori nuli
    if frames.shape[0] == 0:
        M = filters.shape[0] // 2
        return np.zeros(M), np.zeros(M)

    # Inversam filtrele pentru convolutie clasica
    filters_rev = filters[:, ::-1]

    # Filtrare rapida prin produs matricial
    # Rezultat: (numar ferestre x numar filtre)
    responses = np.abs(frames @ filters_rev.T)

    M2 = filters.shape[0] // 2  # jumatate = filtre cos, jumatate = sin

    # Separarea raspunsurilor cos si sin
    cos_mean = responses[:, :M2].mean(axis=0)
    sin_mean = responses[:, M2:].mean(axis=0)
    cos_std  = responses[:, :M2].std(axis=0)
    sin_std  = responses[:, M2:].std(axis=0)

    # Calculam magnitudinea pentru a elimina dependenta de faza
    mag_mean = np.sqrt(cos_mean**2 + sin_mean**2)
    mag_std  = np.sqrt(cos_std**2  + sin_std**2)

    return mag_mean, mag_std


# ================================
# Functia principala ceruta in enunt
# ================================
# Returneaza trasaturi sub forma:
#   matrice D x (2*M)
#
# unde:
# D  = numar de semnale
# M  = numar filtre
# 2M = mean + std pentru fiecare filtru
def get_features(audio_set, fs, M=12, size=1102, hop_ms=12):

    fs = float(fs)

    # calcul pas in esantioane pornind de la milisecunde
    hop = int(round(fs * hop_ms / 1000.0))
    if hop < 1:
        hop = 1

    # generam banca de filtre Gabor
    centers_hz, lengths_hz, cos_f, sin_f = create_gabor_bank(fs, M=M, size=size)

    # concatenam cos si sin -> obtinem 2M filtre
    gabor_bank = np.vstack([cos_f, sin_f])  # dimensiune: (2M, size)

    # numar de semnale
    D = audio_set.shape[0]

    # matricea finala de trasaturi
    features = np.zeros((D, 2 * M))

    for i in range(D):

        # conversie la float pentru stabilitate numerica
        x = np.asarray(audio_set[i], dtype=float)

        # normalizare semnal:
        # eliminam influenta amplitudinii absolute
        if np.std(x) > 1e-12:
            x = (x - np.mean(x)) / (np.std(x) + 1e-12)

        # impartim semnalul in ferestre
        frames = framed_signal(x, frame_len=size, hop=hop)

        # extragem media si deviatia standard pentru fiecare filtru
        mean_g, std_g = compute_features_from_bank(frames, gabor_bank)

        # concatenam pentru a obtine vector 2M
        features[i, :] = np.concatenate([mean_g, std_g])

    # rezultatul final: matrice D x 2M
    return features

from mexican_hat import mexican_hat_filter


# ================================
# Feature extraction folosind Mexican Hat
# ================================
# Aceasta functie realizeaza aceeasi operatie ca get_features,
# dar folosind filtrul Mexican Hat in loc de banci Gabor.

def get_features_mexican(audio_set, fs, size=1102, hop_ms=12):

    fs = float(fs)
    hop = int(round(fs * hop_ms / 1000.0))
    if hop < 1:
        hop = 1

    # generam filtrul Mexican Hat
    sigma = size / 12
    mexican_filter = mexican_hat_filter(size=size, sigma=sigma)

    D = audio_set.shape[0]

    # pentru Mexican Hat avem doar 1 filtru -> mean + std => 2 feature-uri
    features = np.zeros((D, 2))

    for i in range(D):

        x = np.asarray(audio_set[i], dtype=float)

        # normalizare semnal
        if np.std(x) > 1e-12:
            x = (x - np.mean(x)) / (np.std(x) + 1e-12)

        frames = framed_signal(x, frame_len=size, hop=hop)

        if frames.shape[0] == 0:
            features[i, :] = 0
            continue

        # aplicam filtrul Mexican Hat (convolutie)
        responses = np.abs(frames @ mexican_filter[::-1].T)

        # extragem caracterele statistice
        mean_val = np.mean(responses)
        std_val = np.std(responses)

        features[i, :] = [mean_val, std_val]

    return features
