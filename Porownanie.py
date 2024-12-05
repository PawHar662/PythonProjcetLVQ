from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from LVQ import LearningVectorQuantization
from sklearn.decomposition import PCA
import numpy as np


mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist.data, mnist.target.astype(int)
X_mnist = StandardScaler().fit_transform(X_mnist)
X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)


(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = tf.keras.datasets.cifar10.load_data()
X_train_cifar = X_train_cifar.reshape(len(X_train_cifar), -1) / 255.0
X_test_cifar = X_test_cifar.reshape(len(X_test_cifar), -1) / 255.0
y_train_cifar = y_train_cifar.flatten()
y_test_cifar = y_test_cifar.flatten()


pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


lvq = LearningVectorQuantization(liczba_prototypow_na_klase=10, wsp_uczenia=0.001, max_iter=100, random_state=42)
lvq.dopasuj(X_train_pca, y_train)


dokladnosc = lvq.ocen(X_test_pca, y_test)
print("Dokładność na MNIST:", dokladnosc)


pca = PCA(n_components=300)
X_train_pca_cifar = pca.fit_transform(X_train_cifar)
X_test_pca_cifar = pca.transform(X_test_cifar)


scaler = StandardScaler()
X_train_pca_cifar = scaler.fit_transform(X_train_pca_cifar)
X_test_pca_cifar = scaler.transform(X_test_pca_cifar)


lvq = LearningVectorQuantization(liczba_prototypow_na_klase=10, wsp_uczenia=0.001, max_iter=100, random_state=42)
lvq.dopasuj(X_train_pca_cifar, y_train_cifar)


dokladnosc_cifar = lvq.ocen(X_test_pca_cifar, y_test_cifar)
print("Dokładność na CIFAR-10:", dokladnosc_cifar)
