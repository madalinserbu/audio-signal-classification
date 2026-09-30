import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from gabor_filter import create_gabor_bank
import os
import scipy.io


# ================================
# Functie pentru generarea si salvarea graficelor
# ================================
# Genereaza:
#  - forma unui filtru Gabor cos
#  - forma unui filtru Gabor sin
#  - spectrul tuturor filtrelor Gabor
#  - spectrul filtrului Mexican Hat
#
# fs        -> frecventa de esantionare
# id_string -> identificator pentru numele fisierelor
# M         -> numar filtre
# size      -> lungimea fiecarui filtru
def save_gabor_plots(fs, id_string='Serbu_OvidiuMadalin_344C2', M=12, size=1102):

    # cream folderul img daca nu exista
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'img'))
    os.makedirs(img_dir, exist_ok=True)

    print("Saving images in:", img_dir)

    centers, lengths, cos_f, sin_f = create_gabor_bank(fs, M=M, size=size)

    # ================================
    # Afisare filtru Gabor cos
    # ================================
    plt.figure(figsize=(8, 3))
    plt.plot(cos_f[0])
    plt.title('Gabor cos (filter 1)')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'{id_string}_gabor_cos.png'), dpi=150)
    plt.close()


    # ================================
    # Afisare filtru Gabor sin
    # ================================
    plt.figure(figsize=(8, 3))
    plt.plot(sin_f[0])
    plt.title('Gabor sin (filter 1)')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'{id_string}_gabor_sin.png'), dpi=150)
    plt.close()


    # ====================================================
    # SPECTRU FILTRE GABOR
    # ====================================================
    # Afisam spectrul tuturor filtrelor Gabor in acelasi grafic

    plt.figure(figsize=(10, 4))
    Nfft = 4096

    for i in range(M):

        # calcul FFT pentru fiecare filtru
        C = np.abs(fft(cos_f[i], n=Nfft))

        spectrum = C[:600]

        x_axis = np.arange(600)

        plt.plot(x_axis, spectrum, linewidth=1)

    plt.title('Gabor Filters')
    plt.xlabel('Index frecventa')
    plt.ylabel('Amplitudine')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'{id_string}_spectru_filtre.png'), dpi=150)
    plt.close()


    # ====================================================
    # SPECTRU FILTRU MEXICAN HAT
    # ====================================================
    # Parte ceruta in enunt: vizualizarea spectrului unui filtru alternativ

    from mexican_hat import mexican_hat_filter

    plt.figure(figsize=(10, 4))

    # generam filtrul Mexican Hat
    sigma = size / 6
    mex = mexican_hat_filter(size=size, sigma=sigma)

    # calcul FFT pentru Mexican Hat
    C_mex = np.abs(fft(mex, n=Nfft))

    spec_mex = C_mex[:600]
    x_axis = np.arange(600)

    plt.plot(x_axis, spec_mex)
    plt.title('Spectru filtru Mexican Hat')
    plt.xlabel('Index frecventa')
    plt.ylabel('Amplitudine')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'{id_string}_spectru_mexican.png'), dpi=150)
    plt.close()


# ================================
# Executie standalone a scriptului
# ================================
# Cand rulam direct acest fisier, se vor genera automat imaginile.

if __name__ == "__main__":

    # incarcam data.mat doar pentru a prelua fs
    data_path = os.path.join(os.path.dirname(__file__), 'data.mat')
    print("Loading data.mat from:", data_path)

    data = scipy.io.loadmat(data_path)
    fs = float(data['fs'][0, 0])

    save_gabor_plots(fs, id_string='Serbu_OvidiuMadalin_344C2')
