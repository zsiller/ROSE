import pandas as pd
from sklearn.decomposition import PCA


class POD:
    """
    Proper Orthogonal Decomposition (POD) wrapper based on sklearn's PCA.

    This class is used to build a low-rank representation of high-dimensional
    simulation data (or any tabular data) and to later map reduced data back
    to the original space using the PCA model.
    """

    def __init__(self, n_components: int):
        """
        Initialize the POD object.

        Parameters
        ----------
        n_components : int
            Number of principal components (reduced dimensions) to keep.
        """
        self.n_components = n_components
        # Fitted PCA instance (set in fit)
        self.svd = None

    def fit(self, dataframe: pd.DataFrame):
        """
        Fit a PCA model to the provided data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data with samples in rows and features in columns.

        Returns
        -------
        PCA
            The fitted PCA object for further use if needed.
        """
        self.svd = PCA(n_components=self.n_components)
        self.svd.fit(dataframe)

        return self.svd

    def transform(self, dataframe: pd.DataFrame):
        """
        Map reduced data back to the original feature space.

        Note
        ----
        This method currently calls ``inverse_transform``, which assumes that
        ``dataframe`` is already expressed in the reduced POD / PCA space.
        If you want to project high-dimensional data down to the reduced
        space, you would typically call ``self.svd.transform`` instead.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Data expressed in the reduced PCA coordinates.

        Returns
        -------
        ndarray of shape (n_samples, n_original_features)
            Reconstructed data in the original space.
        """
        return self.svd.inverse_transform(dataframe)

