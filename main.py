import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import os

def generate_and_visualize_clusters(n_clusters: int, n_features: int, n_samples: int = 500, random_state: int = 42):
    """
    Генерирует датасет для кластеризации и визуализирует его.
    
    Параметры:
    - n_clusters (int): Количество кластеров (центров).
    - n_features (int): Количество признаков (параметров) для одного элемента.
    - n_samples (int): Общее количество элементов в датасете (по умолчанию 500).
    - random_state (int): Фиксация генератора случайных чисел для воспроизводимости.
    
    Возвращает:
    - X (numpy.ndarray): Сгенерированные данные (признаки).
    - y (numpy.ndarray): Метки кластеров для каждого элемента.
    """
    
    # 1. Генерация датасета
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=5,      # Стандартное отклонение кластеров (можно настроить)
        random_state=random_state
    )

    print(type(X))

    
    # 2. Настройка визуализации
    plt.figure(figsize=(8, 6))
    
    if n_features == 2:
        # Прямая визуализация для 2 признаков
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        plt.xlabel('Признак 1')
        plt.ylabel('Признак 2')
        plt.title(f'Кластеризация: {n_clusters} кластеров, {n_features} признака')
        
    elif n_features == 3:
        # 3D визуализация для 3 признаков
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
        ax.set_zlabel('Признак 3')
        plt.title(f'Кластеризация: {n_clusters} кластеров, {n_features} признака')
        
    else:
        # Для размерности > 3 используем PCA для снижения размерности до 2D
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X)
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        plt.xlabel(f'PCA Компонента 1 ({pca.explained_variance_ratio_[0]*100:.1f}% дисперсии)')
        plt.ylabel(f'PCA Компонента 2 ({pca.explained_variance_ratio_[1]*100:.1f}% дисперсии)')
        plt.title(f'Кластеризация: {n_clusters} кластеров, {n_features} признаков\n(Визуализация через PCA)')
    
    # 3. Общие настройки графика
    plt.colorbar(scatter, label='Номер кластера')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    return X, y


def save_to_csv(X: np.ndarray, y: np.ndarray, filename: str = "cluster_dataset.csv") -> str:
    """
    Сохраняет датасет в CSV-файл.
    
    Параметры:
    - X (numpy.ndarray): Массив признаков (n_samples, n_features).
    - y (numpy.ndarray): Массив меток кластеров (n_samples,).
    - filename (str): Имя файла для сохранения (по умолчанию "cluster_dataset.csv").
    
    Возвращает:
    - str: Абсолютный путь к сохранённому файлу.
    """
    
    # Создаём заголовки столбцов: feature_1, feature_2, ..., cluster
    n_features = X.shape[1]
    feature_columns = [f"feature_{i+1}" for i in range(n_features)]
    columns = feature_columns + ["cluster"]
    
    # Объединяем признаки и метки в один DataFrame
    df = pd.DataFrame(np.column_stack((X, y)), columns=columns)
    
    # Приводим столбец cluster к целочисленному типу для красоты
    df["cluster"] = df["cluster"].astype(int)
    
    # Сохраняем в CSV без индекса строк
    df.to_csv(filename, index=False, float_format="%.6f")
    
    # Возвращаем абсолютный путь для удобства
    abs_path = os.path.abspath(filename)
    return abs_path












# ==========================================
# Пример использования (НАСТРОЙКИ ЗДЕСЬ)
# ==========================================
if __name__ == "__main__":
    # Задайте нужные параметры здесь:
    NUM_CLUSTERS = 4      # Количество кластеров
    NUM_FEATURES = 5      # Количество параметров (признаков) у одного элемента
    NUM_SAMPLES = 800     # Общее количество элементов в датасете
    OUTPUT_FILE = "cluster_dataset.csv"  # Имя выходного CSV-файла
    
    print(f"Генерация датасета: {NUM_SAMPLES} элементов, {NUM_FEATURES} признаков, {NUM_CLUSTERS} кластеров...")
    
    # 1. Генерация и визуализация
    features, labels = generate_and_visualize_clusters(
        n_clusters=NUM_CLUSTERS,
        n_features=NUM_FEATURES,
        n_samples=NUM_SAMPLES,
        random_state=42
    )
    
    # 2. Сохранение в CSV
    saved_path = save_to_csv(features, labels, filename=OUTPUT_FILE)
    
    print(f"Датасет успешно создан! Форма X: {features.shape}, Форма y: {labels.shape}")
    print(f"Файл сохранён по пути: {saved_path}")