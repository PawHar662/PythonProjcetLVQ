import numpy as np

class LearningVectorQuantization:
    def __init__(self, liczba_prototypow_na_klase=1, wsp_uczenia=0.01, max_iter=100, random_state=None):

        self.liczba_prototypow_na_klase = liczba_prototypow_na_klase
        self.wsp_uczenia = wsp_uczenia
        self.max_iter = max_iter
        self.random_state = random_state
        self.prototypy = None
        self.etykiety = None

    def _inicjalizuj_prototypy(self, X, y):

        np.random.seed(self.random_state)
        unikalne_klasy = np.unique(y)
        self.prototypy = []
        self.etykiety = []

        for klasa in unikalne_klasy:
            dane_klasy = X[y == klasa]
            indeksy = np.random.choice(len(dane_klasy), self.liczba_prototypow_na_klase, replace=False)
            self.prototypy.extend(dane_klasy[indeksy])
            self.etykiety.extend([klasa] * self.liczba_prototypow_na_klase)

        self.prototypy = np.array(self.prototypy)
        self.etykiety = np.array(self.etykiety)

    def dopasuj(self, X, y):

        self._inicjalizuj_prototypy(X, y)

        for _ in range(self.max_iter):
            for xi, yi in zip(X, y):
                odleglosci = np.linalg.norm(self.prototypy - xi, axis=1)
                najblizszy_idx = np.argmin(odleglosci)

                if self.etykiety[najblizszy_idx] == yi:
                    self.prototypy[najblizszy_idx] += self.wsp_uczenia * (xi - self.prototypy[najblizszy_idx])
                else:
                    self.prototypy[najblizszy_idx] -= self.wsp_uczenia * (xi - self.prototypy[najblizszy_idx])

    def przewiduj(self, X):

        predykcje = []
        for xi in X:
            odleglosci = np.linalg.norm(self.prototypy - xi, axis=1)
            najblizszy_idx = np.argmin(odleglosci)
            predykcje.append(self.etykiety[najblizszy_idx])
        return np.array(predykcje)

    def ocen(self, X, y):

        predykcje = self.przewiduj(X)
        return np.mean(predykcje == y)

    def pobierz_prototypy(self):

        return self.prototypy, self.etykiety

