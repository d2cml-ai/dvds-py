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

def make_cvgroup(
                size: int,
                n_group: int,
                right: bool = True
) -> np.ndarray:
        split: np.ndarray = np.random.uniform(0, 1, size = size)
        split_quantiles: np.ndarray = np.quantile(split, np.arange(0, 1, 1/n_group))
        cv_groups: np.ndarray = np.digitize(split, bins = split_quantiles, right = right)
        return cv_groups

def make_cvgroup_balanced(Z: NDArray[np.int8], K: int) -> np.ndarray:
        cv_folds: np.ndarray = np.zeros(Z.shape[0])
        cv_folds[Z == 0] = make_cvgroup(
                (Z == 0).sum(), n_group = K, right = False
        )
        cv_folds[Z == 1] = make_cvgroup(
                (Z == 1).sum(), n_group = K, right = True
        )
        return cv_folds

def get_masks(
                Z: NDArray[np.int8],
                cv_groups: NDArray[np.int8],
                fold: int,
                z_value: int,
                odd_train_folds: bool = True,
                semiadaptive: bool = False
) -> tuple[NDArray, NDArray]:
        test_mask: NDArray = cv_groups == fold
        train_mask: NDArray = np.logical_not(test_mask)

        if not semiadaptive:
                nsa_folds: NDArray = (cv_groups - (cv_groups > fold)) % 2 == odd_train_folds
                train_mask = np.logical_and(train_mask, nsa_folds)
        
        train_mask = np.logical_and(train_mask, Z == z_value)
        return train_mask, test_mask

class BinaryConstRegression:

        def fit(
                        self,
                        X: NDArray[Any],
                        y: NDArray[np.int8 | np.float64]
        ) -> None:
                self.train_mean = y.mean()
        
        def predict_proba(
                        self,
                        X: NDArray[Any]
        ) -> NDArray[np.float64]:
                prediction: NDArray[np.float64] = np.array(
                        [self.train_mean] * X.shape[0]
                )
                prediction = np.hstack((1 - prediction, prediction))
                return prediction
        
        def predict(
                        self,
                        X: NDArray
        ) -> NDArray[np.int8]:
                return (self.predict_proba(X)[:, 1] > .5).astype(np.int8)
        
class ContinuousConstRegression:

        def fit(
                        self,
                        X: NDArray[Any],
                        y: NDArray[np.int8 | np.float64]
        ) -> None:
                self.train_mean = y.mean()
        
        def predict(
                        self,
                        X: NDArray[Any]
        ) -> NDArray[np.float64]:
                prediction: NDArray[np.float64] = np.array(
                        [self.train_mean] * X.shape[0]
                )
                return prediction

def cross_fit_propensities(
                X: NDArray[Any],
                Z: NDArray[np.int8],
                cv_groups: NDArray[np.int8],
                propensities_method: Any,
                trim_thresholds: tuple[float, float],
                trim_type: Literal["clip", "drop"] | None,
                normalize: Literal["0", "1"] | None
) -> dict[str, NDArray]:
        prop: NDArray = np.zeros(Z.shape[0])

        for fold in np.unique(cv_groups):
                test_mask: NDArray = cv_groups == fold
                train_mask: NDArray = np.logical_not(test_mask)
                propensities_method.fit(
                        X[train_mask],
                        Z[train_mask]
                )
                prop[test_mask] = propensities_method.predict_proba(X[test_mask])[:, 1].ravel()
        
        indices_for_keeping: np.ndarray = np.array([True] * prop.shape[0])

        if trim_type == "drop":
                indices_for_keeping = np.logical_and(
                        prop > trim_thresholds[0], prop < trim_thresholds[1]
                )
                prop[np.logical_not(indices_for_keeping)] = 0.5
        elif trim_type == "clip":
                prop[prop < trim_thresholds[0]] = trim_thresholds[0]
                prop[prop > trim_thresholds[1]] = trim_thresholds[1]
        
        return {"prop": prop, "keep": indices_for_keeping}

class BootstrapIterationResults:
        
        def __init__(
                        self,
                        X: NDArray,
                        Z: NDArray[np.int8],
                        K: int,
                        propensities_method: Any,
                        refit_propensities: bool,
                        trim_thresholds: tuple[float, float],
                        trim_type: Literal["clip", "drop"] | None,
                        normalize: Literal["0", "1"] | None,
                        prop: dict[str, np.ndarray]
        ) -> None:
                cv_groups: NDArray[np.int8] = make_cvgroup_balanced(Z, K)
                sample_indices: NDArray[np.int64] = np.random.default_rng().choice(
                        Z.shape[0],
                        size = Z.shape[0]
                )

                if refit_propensities:
                        bootstrap_propensities: dict[str, NDArray] = cross_fit_propensities(
                                X[sample_indices], Z[sample_indices],
                                cv_groups[sample_indices], propensities_method,
                                trim_thresholds, trim_type, normalize
                        )
                        bootstrap_propensities_result: NDArray = bootstrap_propensities["prop"]
                else:
                        bootstrap_propensities_result = prop["prop"][sample_indices]
                
                self.props = bootstrap_propensities_result
                self.indices = sample_indices
        
        def get_results(
                        self,
                        lambda_value: float | np.float64,
                        q: NDArray[np.float64 | np.int8],
                        kappa: NDArray[np.float64],
                        y: NDArray[np.int8 | np.float64],
                        Z: NDArray[np.int8],
                        z_value: int,
                        t_value: int,
                        stabilization: bool
        ) -> tuple[np.float64, np.float64, np.float64]:
                q = q[self.indices]
                kappa = kappa[self.indices]
                e: NDArray[np.float64] = self.props
                y = y[self.indices]
                Z = Z[self.indices]
                zz: NDArray[np.int8] = (2 * z_value - 1) * Z + (1 - z_value)
                ee: NDArray[np.float64] = (2 * z_value - 1) * e + (1 - z_value)
                boot_mean_eif: np.float64
                
                if stabilization:
                        ws: NDArray[np.float64] = zz * ((1 / ee) / np.mean(zz / ee) - 1)
                        _hinge: NDArray[np.float64] = q + lambda_value ** np.sign((2 * t_value - 1) * (y - q)) * y - q
                        boot_mean_eif = np.mean(zz * y + (1 - zz) * kappa + ws * (_hinge - kappa))
                else:
                        boot_mean_eif = np.mean(kappa + zz * (q - kappa) / ee + zz * (y - q) * (ee + (1 - ee) * lambda_value ** ((2 * t_value - 1) * np.sign(y - q))))

                return boot_mean_eif, np.mean(y), np.mean(zz)

                

class ConstQuantile:

        def __init__(
                        self,
                        quantile: float
        ) -> None:
                self.tau = quantile

        def fit(
                        self,
                        X: NDArray[Any],
                        y: NDArray[np.int8 | np.float64]
        ) -> None:
                self.train_quantile = np.quantile(
                        y, self.tau
                )
        
        def predict(
                        self,
                        X: NDArray[Any],
        ) -> NDArray[np.float64]:
                prediction: NDArray[np.float64] = np.array(
                        [self.train_quantile] * X.shape[0]
                )
                return prediction
        
class BinaryQuantileExtrapolator:

        def predict(
                        self,
                        quantile: np.float64,
                        muhat: NDArray
        ) -> NDArray[np.int8]:
                self.prediction = (muhat >= (1 - quantile)).astype(np.int8)
                return self.prediction

class BinaryKappaExtrapolator:
        
        def predict(
                        self,
                        quantile: np.float64,
                        muhat: NDArray,
                        quantile_prediction: NDArray[np.int8]
        ) -> NDArray[np.float64]:
                Lambda: np.float64 = np.max(
                        [quantile / (1 - quantile), (1 - quantile) / quantile]
                )
                lambda_exp_sign: np.int8 = np.sign(quantile - 0.5)
                kappa_first_term: NDArray[np.float64] = 1 - Lambda ** (-lambda_exp_sign) * (1 - muhat)
                kappa_second_term: NDArray[np.float64] = Lambda ** (lambda_exp_sign) * muhat
                self.kappa: NDArray[np.float64] = quantile_prediction * kappa_first_term + (1 - quantile_prediction) * kappa_second_term
                return self.kappa

class DVDSStrategy:
        
        def summarize_bootrstrap(
                self,
                bootstrap_estimates: dict[str, NDArray[np.float64]]
        ) -> NDArray[np.float64]:
                bootstrap_summary: NDArray[np.float64]
                bootstrap_summary = np.array(
                        [[np.nan] * 5] * 10
                )

                if len(bootstrap_estimates) < 4:
                        return bootstrap_summary
                
                ci_mean: np.float64
                ci_50: np.float64
                ci_90: np.float64
                ci_95: np.float64
                ci_sd: np.float64
                
                for t in [0, 1]:
                        for z in [0, 1]:
                                ci_mean = np.mean(bootstrap_estimates[f"{z}{t}"][:, 0])
                                ci_50 = np.quantile(bootstrap_estimates[f"{z}{t}"][:, 0], .5)
                                ci_90 = np.quantile(bootstrap_estimates[f"{z}{t}"][:, 0], t * .95 + (1 - t) * .05)
                                ci_95 = np.quantile(bootstrap_estimates[f"{z}{t}"][:, 0], t * .975 + (1 - t) * .025)
                                ci_sd = np.std(bootstrap_estimates[f"{z}{t}"][:, 0], ddof = 1)
                                bootstrap_summary[3 - (2 * z + t)] = [ci_mean, ci_50, ci_90, ci_95, ci_sd]
                                att_atc_bounds: np.ndarray = ((bootstrap_estimates[f"{z}1"][:, 1] - bootstrap_estimates[f"{1 - z}{1 - t}"][:, 0]) / bootstrap_estimates[f"{z}1"][:, 2])
                                ci_mean = np.mean(att_atc_bounds)
                                ci_50 = np.quantile(att_atc_bounds, .5)
                                ci_90 = np.quantile(att_atc_bounds, t * .95 + (1 - t) * .05)
                                ci_95 = np.quantile(att_atc_bounds, t * .975 + (1 - t) * .025)
                                ci_sd = np.std(bootstrap_estimates[f"{z}{t}"][:, 0], ddof = 1)
                                bootstrap_summary[9 - (2 * z + t)] = [ci_mean, ci_50, ci_90, ci_95, ci_sd]
                
                ate_1100_bounds: np.ndarray = bootstrap_estimates["11"][:, 0] - bootstrap_estimates["00"][:, 0]
                ci_mean = np.mean(ate_1100_bounds)
                ci_50 =  np.quantile(ate_1100_bounds, .5)
                ci_90 = np.quantile(ate_1100_bounds, 0.95)
                ci_95 = np.quantile(ate_1100_bounds, 0.975)
                ci_sd = np.std(ate_1100_bounds, ddof = 1)
                bootstrap_summary[4] = [ci_mean, ci_50, ci_90, ci_95, ci_sd]
                ate_1001_bounds: np.ndarray = bootstrap_estimates["10"][:, 0] - bootstrap_estimates["01"][:, 0]
                ci_mean = np.mean(ate_1001_bounds)
                ci_50 =  np.quantile(ate_1001_bounds, .5)
                ci_90 = np.quantile(ate_1001_bounds, 0.95)
                ci_95 = np.quantile(ate_1001_bounds, 0.975)
                ci_sd = np.std(ate_1001_bounds, ddof = 1)
                bootstrap_summary[5] = [ci_mean, ci_50, ci_90, ci_95, ci_sd]
                return bootstrap_summary
        
        def summarize_results(
                        self,
                        lambda_value: np.float64,
                        y: NDArray[np.float64 | np.int8],
                        Z: NDArray[np.int8],
                        influence_function: dict[str, NDArray[np.float64]],
                        standard_errors: dict[str, np.float64],
                        bootstrap_summary: NDArray[np.float64]
        ) -> NDArray:
                att_plus: np.float64 = np.mean(y - influence_function["00"]) / np.mean(Z)
                att_minus: np.float64 = np.mean(y - influence_function["01"]) / np.mean(Z)
                atc_plus: np.float64 = np.mean(y - influence_function["10"]) / np.mean(1 - Z)
                atc_minus: np.float64 = np.mean(y - influence_function["11"]) / np.mean(1 - Z)
                results: NDArray[np.float64 | np.str_]
                estimates_row: NDArray[np.float64]
                standard_errors_row: NDArray[np.float64]
                results = np.array([
                        [lambda_value] * 10,
                        ["1", "1", "0", "0", "ATE", "ATE", "ATT", "ATT", "ATC", "ATC"],
                        ["upper", "lower"] * 5
                ])
                estimates_row = np.array([
                        np.mean(influence_function["11"]),
                        np.mean(influence_function["10"]),
                        np.mean(influence_function["01"]),
                        np.mean(influence_function["00"]),
                        np.mean(influence_function["11"] - influence_function["00"]),
                        np.mean(influence_function["10"] - influence_function["01"]),
                        att_plus,
                        att_minus,
                        atc_plus,
                        atc_minus
                ])
                standard_errors_row = np.array([
                        standard_errors["11"],
                        standard_errors["10"],
                        standard_errors["01"],
                        standard_errors["00"],
                        standard_errors["1100"],
                        standard_errors["1001"],
                        np.std((y - influence_function["00"] - Z * att_plus) / np.mean(Z)) / np.sqrt(y.shape[0]),
                        np.std((y - influence_function["01"] - Z * att_plus) / np.mean(Z)) / np.sqrt(y.shape[0]),
                        np.std((y - influence_function["10"] - (1 - Z) * atc_plus) / np.mean(1 - Z)) / np.sqrt(y.shape[0]),
                        np.std((y - influence_function["11"] - (1 - Z) * atc_plus) / np.mean(1 - Z)) / np.sqrt(y.shape[0])
                ])
                results = np.vstack((results, estimates_row, standard_errors_row))
                results = results.T
                results = np.hstack((results, bootstrap_summary))
                return results
                
        
        def estimate_lambda_bounds(self) -> None:
                pass

        def estimate_bounds(self) -> None:
                pass

class DVDSBinaryStrategy(DVDSStrategy):
        
        def __init__(
                        self,
                        Lambdas: NDArray[np.float64] | list[float],
                        propensities_method: Any,
                        quantiles_method: Any,
                        regression_method: Any,
                        kernel_distance_method: Any, #ForestKernelDistance | KernelDistance
                        muhat_method: Any,
                        K: int,
                        semiadaptive: bool,
                        trim_type: Literal["clip", "drop"] | None,
                        trim_thresholds: tuple[float, float],
                        normalization: Literal["0", "1"] | None,
                        stabilization: bool,
                        **kwargs
        ) -> None:
                if not (
                        isinstance(quantiles_method, BinaryQuantileExtrapolator)
                        and isinstance(regression_method, BinaryKappaExtrapolator)
                ):
                        raise ValueError('"quantiles_method" and "regression_method" must both be of class "BinaryQuantileExtrapolator" and "BinaryKappaExtrapolator" respectively')
                
                if muhat_method is None:
                        raise ValueError('For estimation of binary outcomes, a binary regression method must be provided for "muhat_method"')
                
                self.Lambdas = np.array(Lambdas)
                self.propensities_method = propensities_method
                self.quantiles_method = quantiles_method
                self.regression_method = regression_method
                self.muhat_method = muhat_method
                self.K = K
                self.semiadaptive = semiadaptive
                self.trim_type = trim_type
                self.trim_thresholds = trim_thresholds
                self.normalization = normalization
                self.stabilization = stabilization
                self.kwargs = kwargs
                return
        
        
        def __estimate_lambda_bounds(
                        self,
                        lambda_value: np.float64,
                        y: NDArray[np.float64 | np.int8],
                        Z: NDArray[np.int8],
                        muhat_results: dict[int, NDArray],
                        bootstrap_data: list[BootstrapIterationResults] | None = None,

        ) -> None:
                influence_function: dict[str, NDArray[np.float64]] = {}
                bootstrap_estimations: dict[str, NDArray[np.float64]] = {}
                standard_errors: dict[str, np.float64] = {}
                for z in [0, 1]:
                        for t in [0, 1]:
                                tau: np.float64 = ((1 - t) + t * lambda_value) / (lambda_value + 1)
                                q: NDArray[np.int8] = self.quantiles_method.predict(tau, muhat_results[z]).ravel()
                                _hinge: NDArray[np.float64] = q + lambda_value ** np.sign((2 * t - 1) * (y - q)) * y - q
                                kappa: NDArray[np.float64] = self.regression_method.predict(
                                        tau, muhat_results[z], self.quantiles_method.prediction
                                ).ravel()
                                zz: NDArray[np.int8] = (2 * z - 1) * Z + (1 - z)
                                ee: NDArray[np.float64] = (2 * z - 1) * self.propensities + (1 - z)

                                if self.stabilization:
                                        ws: NDArray[np.float64] = zz * ((1 / ee) / np.mean(zz / ee) - 1)
                                        influence_function[f"{z}{t}"] = zz * y + (1 - zz) * kappa + ws * (_hinge - kappa)
                                else:
                                        influence_function[f"{z}{t}"] = kappa + zz * (q - kappa) / ee + zz * (y - q) * (ee + (1 - ee) * lambda_value ** ((2 * t - 1) * np.sign(y - q))) / ee
                                
                                if bootstrap_data is not None:
                                        bootstrap_start_seed: dict[str, Any] = np.random.get_state()
                                        bootstrap_statistics: NDArray[np.float64] = np.array(
                                                [sample.get_results(
                                                        lambda_value, q, kappa,
                                                        y, Z, z, t, self.stabilization
                                                ) 
                                                 for sample in bootstrap_data]
                                        )
                                        bootstrap_estimations[f"{z}{t}"] = bootstrap_statistics

                                        if self.bootstrap_reset_seed:
                                                np.random.set_state(bootstrap_start_seed)
                                
                                standard_errors[f"{z}{t}"] = np.std(
                                        influence_function[f"{z}{t}"],
                                        ddof = 1
                                ) / np.sqrt(y.shape[0])
                
                standard_errors["1100"] = np.std(
                        influence_function["11"] - influence_function["00"],
                        ddof = 1
                ) / np.sqrt(y.shape[0])
                standard_errors["1001"] = np.std(
                        influence_function["10"] - influence_function["01"],
                        ddof = 1
                ) / np.sqrt(y.shape[0])
                
                bootstrap_summary: NDArray[np.float64] = self.summarize_bootrstrap(bootstrap_estimations)
                lambda_results: NDArray = self.summarize_results(
                        lambda_value, y, Z, influence_function, standard_errors,
                        bootstrap_summary
                )
                
                try:
                        self.results = np.vstack((self.results, lambda_results))
                except ValueError:
                        self.results = lambda_results

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
                self.bootstrap_reset_seed = bootstrap_reset_seed
                
                if X_quant is None:
                        X_quant = X.copy()
                
                if X_kappa is None:
                        X_kappa = X.copy()

                cv_groups: np.ndarray = make_cvgroup_balanced(Z, self.K)
                prop: dict[str, np.ndarray] = cross_fit_propensities(
                        X, Z, cv_groups, self.propensities_method, self.trim_thresholds,
                        self.trim_type, self.normalization #type: ignore
                )
                X = X[prop["keep"]]
                y = y[prop["keep"]]
                self.sample_size = y.shape[0]
                Z = Z[prop["keep"]]
                X_quant = X_quant[prop["keep"]]
                X_kappa = X_kappa[prop["keep"]]
                propensities: NDArray[np.float64] = prop["prop"][prop["keep"]]
                self.propensities = propensities
                cv_groups = cv_groups[prop["keep"]]
                self.cv_groups = cv_groups

                muhat_results: dict[int, NDArray] = {
                        0: np.zeros(y.shape[0]),
                        1: np.zeros(y.shape[0])
                }

                for z_value in [0, 1]:
                        for fold in np.unique(self.cv_groups):
                                train_mask, test_mask = get_masks(
                                        Z, self.cv_groups, fold, z_value, 
                                        odd_train_folds = True,
                                        semiadaptive = self.semiadaptive
                                )
                                self.muhat_method.fit(X[train_mask], y[train_mask])
                                muhat_results[z_value][test_mask] = self.muhat_method.predict_proba(X[test_mask])[:, 1].ravel()
                
                bootstrap_data: list[BootstrapIterationResults] | None = None
                if bootstrap:
                        bootstrap_start_seed: dict[str, Any] = np.random.get_state()
                        bootstrap_data = [
                                BootstrapIterationResults(
                                        X, Z, self.K, self.propensities_method,
                                        bootstrap_refit_propensities,
                                        self.trim_thresholds, self.trim_type, #type: ignore
                                        self.normalization, prop #type:ignore
                                )
                                for _ in range(bootstrap_iterations)
                        ]
                        
                        if self.bootstrap_reset_seed:
                                np.random.set_state(bootstrap_start_seed)
                        
                        del bootstrap_start_seed
                
                self.results = np.array([])
                
                for lambda_value in (pb := tqdm(self.Lambdas)):
                        pb.set_description(f"Estimating bounds for Lambda = {lambda_value}")
                        self.__estimate_lambda_bounds(
                                lambda_value, y, Z, muhat_results, 
                                bootstrap_data
                        )
                
                summary_columns = [
                        "lambda", "estimand", "side", "estimate", "sterr_if",
                        "sterr_boot", "boot_mean", "quantiles50","quantiles90",
                        "quantiles95"
                ]
                self.summary = pd.DataFrame(self.results, columns = summary_columns)
                return
        
class DVDSContinuousStrategy(DVDSStrategy):

        def __init__(
                        self,
                        Lambdas: NDArray[np.float64] | list[float],
                        propensities_method: Any,
                        quantiles_method: Any,
                        regression_method: Any,
                        kernel_distance_method: Any, #ForestKernelDistance | KernelDistance,
                        muhat_method: Any,
                        K: int,
                        semiadaptive: bool,
                        trim_type: Literal["clip", "drop"] | None,
                        trim_thresholds: tuple[float, float],
                        normalization: Literal["0", "1"] | None,
                        stabilization: bool,
                        **kwargs
        ) -> None:
                if (
                        isinstance(quantiles_method, BinaryQuantileExtrapolator | None)
                        or isinstance(regression_method, BinaryKappaExtrapolator| None)
                ):
                        raise ValueError('both "quantiles_method" and "regression_method" must be defined and not be of class "BinaryQuantileExtrapolator" and "BinaryKappaExtrapolator"')
                
                self.Lambdas = np.array(Lambdas)
                self.propensities_method = propensities_method
                self.quantiles_method = quantiles_method
                self.regression_method = regression_method
                self.muhat_method = muhat_method
                self.K = K
                self.semiadaptive = semiadaptive
                self.trim_type = trim_type
                self.trim_thresholds = trim_thresholds
                self.normalization = normalization
                self.stabilization = stabilization
                self.kwargs = kwargs
                return
        
        def __estimate_lambda_bounds(
                        self,
                        lambda_value: np.float64,
                        X: NDArray,
                        y: NDArray[np.float64 | np.int8],
                        Z: NDArray[np.int8],
                        X_quant: NDArray,
                        X_kappa: NDArray,
                        muhat_results: dict[int, NDArray],
                        bootstrap_data: list[BootstrapIterationResults] | None = None
        ) -> None:
                influence_function: dict[str, NDArray[np.float64]] = {}
                bootstrap_estimations: dict[str, NDArray[np.float64]] = {}
                standard_errors: dict[str, np.float64] = {}
                for z in [0, 1]:
                        for t in [0, 1]:
                                tau: np.float64 = ((1 - t) + t * lambda_value) / (lambda_value + 1)
                                q: NDArray[np.float64] = np.zeros(self.sample_size)
                                
                                for fold in np.unique(self.cv_groups):
                                        train_mask, test_mask = get_masks(
                                                Z, self.cv_groups, fold, z,
                                                odd_train_folds = False,
                                                semiadaptive = self.semiadaptive
                                        )
                                        
                                        try:
                                                pretrained_forest: RandomForestQuantileRegressor = self.pretrained_forests[f"{fold}{z}"]
                                                q[test_mask] = pretrained_forest.predict(
                                                        X_quant[test_mask], quantiles = tau
                                                ).ravel()
                                        except AttributeError:
                                                self.quantiles_method.default_quantiles = tau #type:ignore
                                                self.quantiles_method.quantile = tau #type:ignore
                                                self.quantiles_method.fit(
                                                        X_quant[train_mask], 
                                                        y[train_mask]
                                                )
                                                q[test_mask] = self.quantiles_method.predict(X_quant[test_mask]).ravel()
                                
                                kappa: NDArray[np.float64] = np.zeros(self.sample_size)
                                _hinge: NDArray[np.float64]
                                _hinge = q + lambda_value ** (np.sign((2 * t - 1) * (y - q))) * (y - q)

                                for fold in np.unique(self.cv_groups):
                                        train_mask, test_mask = get_masks(
                                                Z, self.cv_groups, fold, z, 
                                                odd_train_folds = False,
                                                semiadaptive = self.semiadaptive
                                        )
                                        self.regression_method.fit(X_kappa[train_mask], _hinge[train_mask])
                                        kappa[test_mask] = self.regression_method.predict(X_kappa[test_mask]).ravel()

                                zz: NDArray[np.int8] = (2 * z - 1) * Z + (1 - z)
                                ee: NDArray[np.float64] = (2 * z - 1) * self.propensities + (1 - z)

                                if self.stabilization:
                                        ws: NDArray[np.float64] = zz * ((1 / ee) / np.mean(zz / ee) - 1)
                                        influence_function[f"{z}{t}"] = zz * y + (1 - zz) * kappa + ws * (_hinge - kappa)
                                else:
                                        influence_function[f"{z}{t}"] = kappa + zz * (q - kappa) / ee + zz * (y - q) * (ee + (1 - ee) * lambda_value ** ((2 * t - 1) * np.sign(y - q))) / ee
                                
                                if bootstrap_data is not None:
                                        bootstrap_start_seed: dict[str, Any] = np.random.get_state()
                                        bootstrap_statistics: NDArray[np.float64] = np.array(
                                                [sample.get_results(
                                                        lambda_value, q, kappa,
                                                        y, Z, z, t, self.stabilization
                                                ) 
                                                 for sample in bootstrap_data]
                                        )
                                        bootstrap_estimations[f"{z}{t}"] = bootstrap_statistics

                                        if self.bootstrap_reset_seed:
                                                np.random.set_state(bootstrap_start_seed)
                                
                                standard_errors[f"{z}{t}"] = np.std(
                                        influence_function[f"{z}{t}"],
                                        ddof = 1
                                ) / np.sqrt(y.shape[0])
                
                standard_errors["1100"] = np.std(
                        influence_function["11"] - influence_function["00"],
                        ddof = 1
                ) / np.sqrt(y.shape[0])
                standard_errors["1001"] = np.std(
                        influence_function["10"] - influence_function["01"],
                        ddof = 1
                ) / np.sqrt(y.shape[0])
                
                bootstrap_summary: NDArray[np.float64] = self.summarize_bootrstrap(bootstrap_estimations)
                lambda_results: NDArray = self.summarize_results(
                        lambda_value, y, Z, influence_function, standard_errors,
                        bootstrap_summary
                )
                
                try:
                        self.results = np.vstack((self.results, lambda_results))
                except ValueError:
                        self.results = lambda_results

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
                self.bootstrap_reset_seed = bootstrap_reset_seed
                
                if X_quant is None:
                        X_quant = X.copy()
                
                if X_kappa is None:
                        X_kappa = X.copy()

                cv_groups: np.ndarray = make_cvgroup_balanced(Z, self.K)
                prop: dict[str, np.ndarray] = cross_fit_propensities(
                        X, Z, cv_groups, self.propensities_method, self.trim_thresholds,
                        self.trim_type, self.normalization #type: ignore
                )
                X = X[prop["keep"]]
                y = y[prop["keep"]]
                self.sample_size = y.shape[0]
                Z = Z[prop["keep"]]
                X_quant = X_quant[prop["keep"]]
                X_kappa = X_kappa[prop["keep"]]
                propensities: NDArray[np.float64] = prop["prop"][prop["keep"]]
                self.propensities = propensities
                cv_groups = cv_groups[prop["keep"]]
                self.cv_groups = cv_groups

                muhat_results: dict[int, NDArray] = {
                        0: np.zeros(y.shape[0]),
                        1: np.zeros(y.shape[0])
                }

                bootstrap_data: list[BootstrapIterationResults] | None = None
                if bootstrap:
                        bootstrap_start_seed: dict[str, Any] = np.random.get_state()
                        bootstrap_data = [
                                BootstrapIterationResults(
                                        X, Z, self.K, self.propensities_method,
                                        bootstrap_refit_propensities,
                                        self.trim_thresholds, self.trim_type, #type: ignore
                                        self.normalization, prop #type:ignore
                                )
                                for _ in range(bootstrap_iterations)
                        ]
                        
                        if self.bootstrap_reset_seed:
                                np.random.set_state(bootstrap_start_seed)
                        
                        del bootstrap_start_seed
                
                if (
                        isinstance(self.quantiles_method, RandomForestQuantileRegressor)
                        and ("reuse_forests" in self.kwargs)
                        and (self.kwargs["reuse_forests"])
                ):
                        pretrained_forests: dict[str, RandomForestQuantileRegressor] = {}
                        all_taus: np.ndarray = np.hstack([
                                1 / (1 + self.Lambdas),
                                self.Lambdas / (1 + self.Lambdas)
                        ])
                        all_taus = np.unique(all_taus)
                        all_taus.sort()

                        for fold in np.unique(self.cv_groups):
                                for z in [0, 1]:
                                        train_mask, test_mask = get_masks(
                                                Z, self.cv_groups, self.K, z, 
                                                semiadaptive = True
                                        )
                                        pretrained_forests[f"{fold}{z}"] = RandomForestQuantileRegressor(
                                                default_quantiles = all_taus.tolist()
                                        ).fit(X_quant[train_mask], y[train_mask])
                        
                        self.pretrained_forests = pretrained_forests
                
                self.results = np.array([])
                
                for lambda_value in (pb := tqdm(self.Lambdas)):
                        pb.set_description(f"Estimating bounds for Lambda = {lambda_value}")
                        self.__estimate_lambda_bounds(
                                lambda_value, X, y, Z, X_quant, X_kappa,
                                muhat_results, bootstrap_data
                        )
                
                summary_columns = [
                        "lambda", "estimand", "side", "estimate", "sterr_if",
                        "sterr_boot", "boot_mean", "quantiles50","quantiles90",
                        "quantiles95"
                ]
                self.summary = pd.DataFrame(self.results, columns = summary_columns)
                return 
