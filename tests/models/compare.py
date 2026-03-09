import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statsmodels.api as sm

def evaluate_models(soaps, dfs, model_list, *,
                    i=1, N_fit=100, n_splits=5,
                    axes=None,
                    kcal_mol=True, plot=True, energy_key="E"):
    """
    Evaluate multiple models on SOAP + energy datasets using train/test or K-Fold CV.
    
    Uses statsmodels to compute R², RMSE, MAE, and returns a summary DataFrame.
    """
    df_ = dfs[i].copy()
    df_[energy_key] -= df_[energy_key].min()
    soaps_ = soaps[i]

    def compute_stats(y_true, y_pred, kcal_mol=True):
        residuals = y_true - y_pred
    
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Fit statsmodels for diagnostics
        model = sm.OLS(y_true, sm.add_constant(y_pred)).fit()
        
        if kcal_mol:
            mae *= 627.5
            rmse *= 627.5
    
        stats = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": model.rsquared,
            "Adj_R2": model.rsquared_adj,
            "AIC": model.aic,
            "BIC": model.bic,
            "Skewness": pd.Series(residuals).skew(),
            "Kurtosis": pd.Series(residuals).kurtosis(),
            "Durbin_Watson": sm.stats.stattools.durbin_watson(residuals),
            "Max_Error": np.max(np.abs(residuals)),
            "Min_Error": np.min(np.abs(residuals)),
            "Condition_Number": model.condition_number
        }
    
        if kcal_mol:
            stats["Max_Error"] *= 627.5
            stats["Min_Error"] *= 627.5
    
        return stats

    def plot_preds(y_true, y_pred, title, c="k", ax=None):
        from cmap import Colormap
        cm = Colormap('crameri:berlin')  # case insensitive
        _cmap = cm.to_matplotlib()
        # plt.figure()
        ax = plt.gca() if ax is None else ax
        ax.set_title(title)
        ax.scatter(y_pred, y_true, alpha=0.3, c=c, cmap=_cmap)
        ax.set_aspect("equal")
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, c="k")
        ax.set_xlim(0, 0.03)
        ax.set_ylim(0, 0.03)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # plt.show()

    def to_str(X): return [str(x) for x in X]

    results = []

    df_train = df_.sample(N_fit, weights=(df_[energy_key] - df_[energy_key].median())**2, random_state=42)
    df_test = df_.drop(df_train.index)

    X_train = np.array([soaps_[j] for j in df_train.index])
    y_train = df_train[energy_key].values
    X_test = np.array([soaps_[j] for j in df_test.index])
    y_test = df_test[energy_key].values

    for i, (name, model) in enumerate(model_list):
        if getattr(model, "can_handle_strings", False):
            X_train_ = to_str(X_train)
            X_test_ = to_str(X_test)
        else:
            X_train_ = X_train
            X_test_ = X_test

        model.fit(X_train_, y_train)
        y_pred = model.predict(X_test_)
        stats = compute_stats(y_test, y_pred)
        stats.update({"Model": name, "Split": "KFold" or f"Train={N_fit}"})
        results.append(stats)
        if plot:
            
            ax = axes[i] if axes else None
            plot_preds(y_test, y_pred, f"{name}", ax=ax, c=df_test["com_dists"])

    return pd.DataFrame(results)


# In[462]:


import molzip
from molzip.regressor import ZipRegressor, regress


# In[463]:


import gzip
import multiprocessing
from typing import Any, Iterable
from functools import partial
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def fig2img(fig,dpi=9):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, metadata=None, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img



class _ZipRegressor(object):
    def __init__(self, k=1) -> None:
        super().__init__()
        # self.compressor = compressor
        self.fitted = False
        self.k = k
        self.top_k_values = None
        self.top_k_dists = None
        self.n_props = None

    def fit_predict(
        self,
        X_train: Iterable[str],
        y_train: Iterable,
        X: Iterable[str],
        k: int = 25,
     ) -> np.ndarray:
        preds = []

        y_train = np.array(y_train)

        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, axis=1)

        cpu_count = multiprocessing.cpu_count()

        with multiprocessing.Pool(cpu_count) as p:
            preds = p.map(
                partial(
                    regress,
                    X_train=X_train,
                    y_train=y_train,
                    k=k,
                ),
                X,
            )

        return np.array(preds)

    def fit (
        self,
        X_train: Iterable[str],
        y_train: Iterable,
    ) -> np.ndarray:
        preds = []
        self.X_train = [str(_) for _ in X_train]
        self.y_train = np.array(y_train, dtype=float)
        return self.fit_predict(self.X_train, self.y_train, self.X_train, k=self.k).flatten()

    def predict (
        self,
        X: Iterable[str],
        k: int = 25,
    ) -> np.ndarray:
        return self.fit_predict(self.X_train, self.y_train, [str(_) for _ in X], k=self.k).flatten()



 


# In[464]:


# SGDRegressor?


# In[465]:


from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

k = 50

model_list = [
    ("gzKNN ($k="+str(2)+"$)", _ZipRegressor(k=2)),
    ("gzKNN ($k="+str(10)+"$)", _ZipRegressor(k=10)), 
    ("gzKNN ($k="+str(20)+"$)", _ZipRegressor(k=20)),
    ("gzKNN ($k="+str(50)+"$)", _ZipRegressor(k=50)),
    ("gzKNN ($k="+str(100)+"$)", _ZipRegressor(k=100)),
    ("gzKNN ($k="+str(200)+"$)", _ZipRegressor(k=200)),
    # 🔹 Linear and regularized models
    ("LinearRegression", make_pipeline(StandardScaler(), LinearRegression())),
    ("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("Lasso", make_pipeline(StandardScaler(), Lasso(alpha=0.0))),
    ("ElasticNet", make_pipeline(StandardScaler(), ElasticNet(alpha=0.0, l1_ratio=0.5))),
    # ("SGD", make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))),

    # 🔸 Kernel-based models
    ("SVR-RBF", make_pipeline(StandardScaler(), SVR(kernel='rbf', C=0.01, epsilon=0.0))),
    ("KernelRidge", make_pipeline(StandardScaler(), KernelRidge(alpha=0.0, kernel='rbf'))),

    # 🔹 Tree-based and ensemble models
    ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)),
    ("AdaBoost", AdaBoostRegressor(n_estimators=100)),

    # 🔸 Neural net
    ("MLP", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,64),
                                        early_stopping=True,
                                        max_iter=100000, random_state=42))),

    # 🔹 Gaussian process
    ("GPR", make_pipeline(StandardScaler(), 
        GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0), alpha=1e-6, random_state=42))),
]



# In[488]:


import matplotlib.pyplot as plt

def create_model_mosaic(model_names, layout=None, figsize=(14, 8)):
    """
    Create a subplot mosaic with model names as semantic keys.
    
    Parameters:
    - model_names: list of strings, names of models (used as subplot keys)
    - layout: list of lists of strings, custom layout (optional)
    - figsize: size of the figure
    
    Returns:
    - fig: the matplotlib Figure
    - axes: dictionary of Axes objects keyed by model name
    """
    # If no custom layout is provided, build a default grid
    if layout is None:
        n_cols = 3
        rows = [
            model_names[i:i + n_cols]
            for i in range(0, len(model_names), n_cols)
        ]
    else:
        rows = layout
    
    fig, axes = plt.subplot_mosaic(rows, figsize=figsize, 
                                   # sharex=True, sharey=True
                                  )
    return fig, axes


# In[467]:


model_names = [_[0] for _ in model_list]
model_names


# In[468]:


figure_layout = [
['gzKNN ($k=2$)',
 'gzKNN ($k=10$)',
 'gzKNN ($k=20$)',
 'gzKNN ($k=50$)',
 'gzKNN ($k=100$)',
 'gzKNN ($k=200$)',],
    [ 'LinearRegression',
 'Ridge',
 'Lasso',
 'ElasticNet',
 'SVR-RBF',
 'KernelRidge',
],
    [ 'RandomForest',
     'GradientBoosting',
 'AdaBoost',
 'MLP',
 'GPR',
    "X"],
]