import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn. linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.svm import SVC, SVR
from quantile_forest import RandomForestQuantileRegressor
from tqdm import tqdm
import pandas as pd
from strategies import BinaryConstRegression, BinaryKappaExtrapolator, BinaryQuantileExtrapolator, DVDSBinaryStrategy, DVDSContinuousStrategy

class DVDSBoundsEstimator:
        """
        Context class for estimation of doubly-valid, doubly-sharp bounds.
        # Parameters:
                - `Lambdas: NDArray[np.float64] | list[float]`: List or 1-D array of values for the magnitude of the effect of a hypothetical confounder.
                - `outcome_estimation_type: Literal["binary", "continuous", "kernel"] = "binary"`: Estimation strategy for the bounds. For the estimation of quantiles and transformed outcomes:
                        * "binary" uses binary extrapolation through the `BinaryQuantileExtrapolator` and the `BinaryKappaExtrapolator` classes;
                        * "continuous" uses continuous variable regression and prediction methods with a `.fit()` and `.predict()` method, as in the scikit-learn package; and
                        * "kernel" uses kernel weights, either from a random forest estimator or from the features' covariance matrix
                - `propensities_method: Any = LogisticRegression()`: Binary regressor for treatment propensity estimation.
                - `quantiles_method: Any = BinaryQuantileExtrapolator()`: Regressor for estimating outcome quantiles.
                - `regression_method:  Any = BinaryKappaExtrapolator()`: Regressor for estimating transformed outcomes.
                - `kernel_distance_method: Any = None`: Method for computing kernel weights. Only used when `outcome_estimation_type = "kernel"`.
                - `muhat_method: Any = LogisticRegression()`: Regressor for cross-fit predictions of the outcome. Only used when `outcome_estimation_type = "binary"`.
                - `K: int = 5`: Number of folds for cross-fitting
                - `semiadaptive: bool = False`: Whether to use the same folds to fit the quantiles and the transformed outcomes. Requires K >= 5 and odd.
                - `trim_type: Literal["clip", "drop"] | None = "clip"`: Method for trimming the estimated propensities:
                        * "clip" winzorizes the propensities,
                        * "drop" drops the observations with propensities outside of the interval
                - `trim_thresholds: tuple[float, float] = (0.01, 0.99)`: Interval for the trimming procedure.
                - `normalization: Literal["0", "1"] | None = None`: Normalize propensities when "0" or "1". 
                - `stabilization: bool = False`: # TODO
        """

        def __init__(
                        self,
                        Lambdas: NDArray[np.float64] | list[float],
                        outcome_estimation_type: Literal["binary", "continuous", "kernel"] = "binary",
                        propensities_method: Any = LogisticRegression(),
                        quantiles_method: Any = BinaryQuantileExtrapolator(),
                        regression_method:  Any = BinaryKappaExtrapolator(),
                        kernel_distance_method: Any = None, #ForestKernelDistance | KernelDistance
                        muhat_method: Any = LogisticRegression(),
                        K: int = 5,
                        semiadaptive: bool = False,
                        trim_type: Literal["clip", "drop"] | None = "clip",
                        trim_thresholds: tuple[float, float] = (0.01, 0.99),
                        normalization: Literal["0", "1"] | None = None,
                        stabilization: bool = False
        ) -> None:
                if (not semiadaptive) and ((K < 5) or (K % 2 == 0)):
                        raise ValueError("For non-semiadaptive cross-fitting, K must be greater than or equal to 5, and it must be odd")
                
                if trim_thresholds[0] > trim_thresholds[1]:
                        raise ValueError("Thresholds for trimming must be in ascending order")
                
                self.Lambdas = Lambdas
                self.outcome_estimation_type = outcome_estimation_type
                self.propensities_method = propensities_method
                self.quantiles_method = quantiles_method
                self.regression_method = regression_method
                self.kernel_distance_method = kernel_distance_method
                self.muhat_method = muhat_method
                self.K = K
                self.semiadaptive = semiadaptive
                self.trim_type = trim_type
                self.trim_thresholds = trim_thresholds
                self.normalization = normalization
                self.stabilization = stabilization

                if outcome_estimation_type == "binary":
                        self.estimation_strategy = DVDSBinaryStrategy(
                                self.Lambdas,
                                self.propensities_method,
                                self.quantiles_method, #type:ignore
                                self.regression_method, #type:ignore
                                self.kernel_distance_method,
                                self.muhat_method, #type:ignore
                                self.K,
                                self.semiadaptive,
                                self.trim_type,
                                self.trim_thresholds,
                                self.normalization,
                                self.stabilization
                        )
                elif outcome_estimation_type == "continuous":
                        self.estimation_strategy = DVDSContinuousStrategy(
                                self.Lambdas,
                                self.propensities_method,
                                self.quantiles_method, #type:ignore
                                self.regression_method, #type:ignore
                                self.kernel_distance_method,
                                self.muhat_method, #type:ignore
                                self.K,
                                self.semiadaptive,
                                self.trim_type,
                                self.trim_thresholds,
                                self.normalization,
                                self.stabilization
                        )
                elif outcome_estimation_type == "kernel":
                        pass # DVDSKernelStrategy
                else:
                        raise ValueError("Argument outcome_estimation_type \
                                         must be one of 'binary' 'continuous' \
                                         or 'kernel'")
                
                return
        
        def estimate_bounds(
                        self,
                        X: NDArray[Any],
                        y: NDArray[np.float64 | np.int8],
                        Z: NDArray[np.int8],
                        X_quant: NDArray[Any] | None = None,
                        X_kappa: NDArray[Any] | None = None,
                        bootstrap: bool = False,
                        bootstrap_refit_propensities: bool = False,
                        bootstrap_iterations: int = 500,
                        bootstrap_reset_seed: bool = True
        ) -> None:
                self.estimation_strategy.estimate_bounds(
                        X, y, Z, X_quant, X_kappa, bootstrap,
                        bootstrap_refit_propensities, bootstrap_iterations,
                        bootstrap_reset_seed
                )
                self.results = self.estimation_strategy.results
                self.summary = self.estimation_strategy.summary

