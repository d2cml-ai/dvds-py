from tqdm import tqdm
import seaborn as sb
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def phi_continuous(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):