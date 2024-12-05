import unittest
import numpy as np
from sklearn.datasets import make_classification
from LVQ import LearningVectorQuantization

class TestLearningVectorQuantization(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=20, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=42
        )
        self.model = LearningVectorQuantization(
            liczba_prototypow_na_klase=1, wsp_uczenia=0.1, max_iter=50, random_state=42
        )

    def test_dopasowanie(self):
        self.model.dopasuj(self.X, self.y)
        prototypy, etykiety = self.model.pobierz_prototypy()
        self.assertEqual(len(prototypy), 2)
        self.assertTrue(np.all(np.isin(etykiety, np.unique(self.y))))

    def test_przewidywanie(self):
        self.model.dopasuj(self.X, self.y)
        predykcje = self.model.przewiduj(self.X)
        self.assertEqual(len(predykcje), len(self.y))
        self.assertTrue(np.all(np.isin(predykcje, np.unique(self.y))))

    def test_ocena(self):
        self.model.dopasuj(self.X, self.y)
        dokladnosc = self.model.ocen(self.X, self.y)
        self.assertGreaterEqual(dokladnosc, 0.0)
        self.assertLessEqual(dokladnosc, 1.0)

if __name__ == "__main__":
    unittest.main()

