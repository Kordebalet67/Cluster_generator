import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Получение датасета: 
# 1. запустить заново генерацию из файла
# 2. загрузить из файла

from generator import generate_and_visualize_clusters


# Обучение модели 
def train_model(X, y, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)    

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Размер X_train: {X_train.shape[0]}')
    print(f'Размер y_train: {y_train.shape[0]}')
    print(f'Размер X_test: {X_test.shape[0]}')
    print(f'Размер y_test: {y_test.shape[0]}')
    print(f'Точность модели: {accuracy}')
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, X_train, X_test, y_train, y_test, y_pred

# Визуализация решения
def plot_results(model, X, y, n_features, random_state=42):
    if n_features == 2:
        X_2d = X
        pca = None
    else:
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X)

    h = 0.1
    x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
    y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if pca is not None:
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_full = pca.inverse_transform(grid_2d)
        Z = model.predict(grid_full)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(10,10))
    plt.contourf(xx, yy, Z, alpha = 0.3, cmap = 'viridis')
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='viridis', edgecolors='k', alpha=0.8, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Ось 1'+('(PCA)' if pca else ''))
    plt.ylabel('Ось 2'+('(PCA)' if pca else ''))
    plt.tight_layout()
    plt.show()

# Сохранение модели
def save_model(model, filename='classifier_model.joblib'):
    import joblib
    joblib.dump(model, filename)
    return os.path.abspath(filename)

# Определение кластера для нового объекта
def recognize_cluster(model, X_train, random_state=42):
    rng = np.random.default_rng(random_state)
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    new_point = rng.uniform(mins, maxs).reshape(1, -1)
    prediction = model.predict(new_point)[0]
    probabilities = model.predict_proba(new_point)[0]

    print(f'Координаты точки: {new_point[0].round(3)}')
    print(f'Кластер: {prediction}')
    for cls, prob in enumerate(probabilities): 
        print(f'Кластер {cls}: Вероятность: {prob}')
    return new_point, prediction

if __name__ == '__main__':
    # generate
    X, y = generate_and_visualize_clusters(
        n_clusters=5,
        n_features=4,
        n_samples=1_000,
        random_state=42
    )

    # train
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    # visual
    plot_results(y_test, y_pred)
    # save
    save_model(model)
    # usage
    loaded_model = model#load_model('')
    test_pred = loaded_model.predict(X_test[:3])
    print(test_pred)