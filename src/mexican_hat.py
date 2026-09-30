import numpy as np


# ================================
# Filtru Mexican Hat (Ricker wavelet)
# ================================
# Este derivata a doua a unei functii Gaussiene si are comportament de tip band-pass.
#
# size  -> lungimea filtrului (numar de esantioane)
# sigma -> controleaza latimea filtrului (rezolutia in frecventa)
def mexican_hat_filter(size, sigma):

    # Stabilim sigma proportional cu dimensiunea ferestrei,
    # pentru a obtine un filtru stabil si coerent vizual.
    sigma = size / 12

    n = np.arange(size)

    # centrul ferestrei
    mu = (size - 1) / 2.0

    # deplasare fata de centru
    x = (n - mu)

    # Formula matematica a wavelet-ului Mexican Hat:
    hh = (1.0 - (x ** 2) / (sigma ** 2)) * np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    hh = hh - np.mean(hh)

    # Normalizare:
    # previne influenta amplitudinii
    hh = hh / (np.linalg.norm(hh) + 1e-12)

    return hh
