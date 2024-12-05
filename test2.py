from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf


mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist.data, mnist.target.astype(int)
X_mnist = StandardScaler().fit_transform(X_mnist)  # Normalizacja danych
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)


pca_mnist = PCA(n_components=50)
X_train_pca_mnist = pca_mnist.fit_transform(X_train_mnist)
X_test_pca_mnist = pca_mnist.transform(X_test_mnist)


(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = tf.keras.datasets.cifar10.load_data()
X_train_cifar = X_train_cifar.reshape(len(X_train_cifar), -1) / 255.0  # Spłaszczenie i normalizacja
X_test_cifar = X_test_cifar.reshape(len(X_test_cifar), -1) / 255.0
y_train_cifar = y_train_cifar.flatten()
y_test_cifar = y_test_cifar.flatten()


pca_cifar = PCA(n_components=300)
X_train_pca_cifar = pca_cifar.fit_transform(X_train_cifar)
X_test_pca_cifar = pca_cifar.transform(X_test_cifar)


scaler_cifar = StandardScaler()
X_train_pca_cifar = scaler_cifar.fit_transform(X_train_pca_cifar)
X_test_pca_cifar = scaler_cifar.transform(X_test_pca_cifar)


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Dokładność {model_name}: {accuracy:.4f}")
    return accuracy


models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(max_iter=300)
}


print("Porównanie na zbiorze MNIST:")
mnist_results = {}
for name, model in models.items():
    mnist_results[name] = evaluate_model(model, X_train_pca_mnist, y_train_mnist, X_test_pca_mnist, y_test_mnist, name)


print("\nPorównanie na zbiorze CIFAR-10:")
cifar_results = {}
for name, model in models.items():
    cifar_results[name] = evaluate_model(model, X_train_pca_cifar, y_train_cifar, X_test_pca_cifar, y_test_cifar, name)


print("\nWyniki na MNIST:")
for name, accuracy in mnist_results.items():
    print(f"{name}: {accuracy:.4f}")

print("\nWyniki na CIFAR-10:")
for name, accuracy in cifar_results.items():
    print(f"{name}: {accuracy:.4f}")
