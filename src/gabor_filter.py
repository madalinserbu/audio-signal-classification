import numpy as np


# ================================
# Conversie frecventa Hz -> scala Mel
# ================================
def mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


# ================================
# Conversie inversa Mel -> Hz
# ================================
def inv_mel(m):
    return 700.0 * (np.exp(m / 1127.0) - 1.0)


# ================================
# Filtru Gabor individual
# ================================
# size  -> lungimea filtrului (numar de esantioane)
# sigma -> controleaza latimea ferestrei Gaussiene (rezolutia temporala)
# freq  -> frecventa centrala a filtrului (normalizata)
def gabor_filter(size, sigma, freq):

    n = np.arange(size)

    # centrul ferestrei pentru simetrie
    mu = (size - 1) / 2.0

    # Functia Gaussiana
    # asigura localizare temporala a filtrului
    gauss = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * \
            np.exp(-((n - mu) ** 2) / (2.0 * sigma ** 2))

    # Modulare cu cos si sin -> localizare in frecventa
    # Astfel filtrul raspunde doar la o banda de frecventa
    cos_h = gauss * np.cos(2.0 * np.pi * freq * (n - mu))
    sin_h = gauss * np.sin(2.0 * np.pi * freq * (n - mu))

    # Normalizare:
    # previne influenta amplitudinii filtrului asupra rezultatelor
    cos_h = cos_h / (np.linalg.norm(cos_h) + 1e-12)
    sin_h = sin_h / (np.linalg.norm(sin_h) + 1e-12)

    return cos_h, sin_h


# Creeaza setul de M filtre distribuite pe scala Mel intre 0 si fs/2.
# fs   -> frecventa de esantionare
# M    -> numarul de filtre (implicit 12)
# size -> lungimea fiecarui filtru
def create_gabor_bank(fs, M=12, size=1102):

    # limitele domeniului Mel
    mel_low = mel(0.0)
    mel_high = mel(fs / 2.0)

    # impartire uniforma pe scala Mel
    mel_boundaries = np.linspace(mel_low, mel_high, M + 1)

    centers_hz = np.zeros(M)   # frecvente centrale
    lengths_hz = np.zeros(M)   # latimi de banda

    # matrici care contin filtrele
    cos_filters = np.zeros((M, size))
    sin_filters = np.zeros((M, size))

    for i in range(M):

        # limitele benzii i pe scala Mel
        mel_a = mel_boundaries[i]
        mel_b = mel_boundaries[i + 1]

        # centrul benzii
        mel_center = 0.5 * (mel_a + mel_b)

        # conversie in frecventa reala
        c_i = inv_mel(mel_center)

        # latimea benzii in Hz
        l_i = inv_mel(mel_b) - inv_mel(mel_a)

        centers_hz[i] = c_i
        lengths_hz[i] = l_i

        # frecventa normalizata (raportata la fs)
        fi = c_i / fs

        # sigma stabileste rezolutia filtrului:
        # banda mai ingusta -> filtru mai selectiv
        sigma_i = (fs / l_i) if (l_i > 1e-6) else (size / 6.0)

        # generam filtrul Gabor pentru aceasta banda
        cos_h, sin_h = gabor_filter(size=size, sigma=sigma_i, freq=fi)

        cos_filters[i, :] = cos_h
        sin_filters[i, :] = sin_h

    return centers_hz, lengths_hz, cos_filters, sin_filters
