import os
import time

import h5py
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ROWS_PER_FILE = 61
N_FEATURES = 256
N_COMPONENTS = 2


def list_h5_files(data_dir: str) -> list[str]:
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".h5")
    )


def load_batch(file_path: str) -> np.ndarray:
    with h5py.File(file_path, "r") as handle:
        batch = handle["rho"][:]
    if batch.shape != (ROWS_PER_FILE, N_FEATURES):
        raise ValueError(
            f"{file_path}: expected shape {(ROWS_PER_FILE, N_FEATURES)}, got {batch.shape}"
        )
    return batch.astype(np.float32, copy=False)


def load_all_data(file_paths: list[str]) -> np.ndarray:
    if not file_paths:
        raise ValueError(f"No .h5 files found in {DATA_DIR}")

    data = np.empty((ROWS_PER_FILE * len(file_paths), N_FEATURES), dtype=np.float32)
    row_offset = 0
    for file_path in file_paths:
        batch = load_batch(file_path)
        data[row_offset : row_offset + ROWS_PER_FILE] = batch
        row_offset += ROWS_PER_FILE
    return data


def fit_batch_pca(data: np.ndarray, n_components: int) -> PCA:
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca


def fit_incremental_pca(file_paths: list[str], n_components: int) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=n_components)
    for file_path in file_paths:
        ipca.partial_fit(load_batch(file_path))
    return ipca


def align_component_signs(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    aligned = candidate.copy()
    for idx in range(candidate.shape[0]):
        if np.dot(reference[idx], candidate[idx]) < 0:
            aligned[idx] *= -1
    return aligned


def compare_pca_models(
    batch_pca: PCA,
    incremental_pca: IncrementalPCA,
    data: np.ndarray,
    *,
    batch_load_seconds: float,
    batch_fit_seconds: float,
    incremental_seconds: float,
) -> None:
    batch_components = batch_pca.components_
    incremental_components = align_component_signs(
        batch_components,
        incremental_pca.components_,
    )

    batch_scores = batch_pca.transform(data)
    incremental_scores = incremental_pca.transform(data)

    batch_recon = batch_pca.inverse_transform(batch_scores)
    incremental_recon = incremental_pca.inverse_transform(incremental_scores)

    print("PCA comparison (batch vs incremental)")
    print("=" * 60)
    print(f"Samples: {data.shape[0]}, features: {data.shape[1]}, components: {N_COMPONENTS}")
    print(f"Files used: {data.shape[0] // ROWS_PER_FILE}")
    print()

    print("Timing:")
    print(f"  batch load:        {batch_load_seconds:.4f}s")
    print(f"  batch fit:         {batch_fit_seconds:.4f}s")
    print(f"  batch total:       {batch_load_seconds + batch_fit_seconds:.4f}s")
    print(f"  incremental total: {incremental_seconds:.4f}s (load + partial_fit per file)")
    print()

    print("Explained variance ratio:")
    for idx, (batch_ratio, incremental_ratio) in enumerate(
        zip(batch_pca.explained_variance_ratio_, incremental_pca.explained_variance_ratio_),
        start=1,
    ):
        diff = abs(batch_ratio - incremental_ratio)
        print(
            f"  PC{idx}: batch={batch_ratio:.8f}, "
            f"incremental={incremental_ratio:.8f}, diff={diff:.2e}"
        )
    print()

    print("Principal component cosine similarity (sign-aligned):")
    for idx, (batch_vec, incremental_vec) in enumerate(
        zip(batch_components, incremental_components),
        start=1,
    ):
        cosine = np.dot(batch_vec, incremental_vec) / (
            np.linalg.norm(batch_vec) * np.linalg.norm(incremental_vec)
        )
        max_abs_diff = np.max(np.abs(batch_vec - incremental_vec))
        print(f"  PC{idx}: cosine={cosine:.12f}, max|component diff|={max_abs_diff:.2e}")
    print()

    score_diff = batch_scores - incremental_scores
    print("Transformed coordinates:")
    print(f"  max|score diff|={np.max(np.abs(score_diff)):.2e}")
    print(f"  mean|score diff|={np.mean(np.abs(score_diff)):.2e}")
    print()

    batch_recon_error = np.mean((data - batch_recon) ** 2)
    incremental_recon_error = np.mean((data - incremental_recon) ** 2)
    cross_recon_error = np.mean((data - incremental_pca.inverse_transform(batch_scores)) ** 2)

    print("Mean squared reconstruction error:")
    print(f"  batch PCA on batch scores:           {batch_recon_error:.2e}")
    print(f"  incremental PCA on incremental scores: {incremental_recon_error:.2e}")
    print(f"  incremental PCA on batch scores:       {cross_recon_error:.2e}")
    print()

    print("Sample row 0 scores:")
    print(f"  batch:       {batch_scores[0]}")
    print(f"  incremental: {incremental_scores[0]}")


def main() -> None:
    file_paths = list_h5_files(DATA_DIR)
    print(f"Found {len(file_paths)} files in {DATA_DIR}")

    load_start = time.perf_counter()
    data = load_all_data(file_paths)
    batch_load_seconds = time.perf_counter() - load_start
    print(f"Loaded full dataset into memory for comparison: {data.shape}")

    batch_fit_start = time.perf_counter()
    batch_pca = fit_batch_pca(data, N_COMPONENTS)
    batch_fit_seconds = time.perf_counter() - batch_fit_start

    incremental_start = time.perf_counter()
    incremental_pca = fit_incremental_pca(file_paths, N_COMPONENTS)
    incremental_seconds = time.perf_counter() - incremental_start

    compare_pca_models(
        batch_pca,
        incremental_pca,
        data,
        batch_load_seconds=batch_load_seconds,
        batch_fit_seconds=batch_fit_seconds,
        incremental_seconds=incremental_seconds,
    )


if __name__ == "__main__":
    main()
