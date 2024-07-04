from ctypes import ArgumentError
from math import inf, sqrt
from typing import Any, Callable, Literal, Union
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn. linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.svm import SVC, SVR
from quantile_forest import RandomForestQuantileRegressor
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from formulaic import ModelMatrix, model_matrix
import warnings
from tqdm import tqdm

def make_cvgroup_balanced(data: pd.DataFrame, K: int, z_column: str) -> np.ndarray:
        data_indices: np.ndarray = np.array([i for i in range(data.shape[0])])
        kfold_splitter: KFold = KFold(n_splits = K, shuffle = True)
        zeros_indices: np.ndarray = data_indices[data[z_column] == 0]
        ones_indices: np.ndarray = data_indices[data[z_column] == 1]
        zeros_cv: list[np.ndarray] = [zeros_indices[split[1]] for split in kfold_splitter.split(zeros_indices)]
        ones_cv: list[np.ndarray] = [ones_indices[split[1]] for split in kfold_splitter.split(ones_indices)]
        fold_indices: list[np.ndarray] = [
                np.hstack((array_zeros, array_ones)) 
                for array_zeros, array_ones in zip(zeros_cv, ones_cv)
        ]
        cv_folds: np.ndarray = np.zeros(data.shape[0])
        for index, fold in enumerate(fold_indices):
                cv_folds[fold] = index
        return cv_folds

def get_model_matrix(
                data: pd.DataFrame,
                response: str,
                form_x: str
) -> tuple[np.ndarray, np.ndarray]:
        formula: str = response + "~" + form_x
        matrices: tuple[ModelMatrix, ModelMatrix] = model_matrix(formula, data)
        y: np.ndarray = matrices.lhs.to_numpy().reshape(len(matrices.lhs))
        X: np.ndarray = matrices.rhs.to_numpy()
        return X, y

def cross_fit_propensities(
                data: pd.DataFrame,
                cvgroup: np.ndarray,
                formula_x: str,
                z_column: str,
                method_prop: Callable,
                trim_thresholds: tuple[float, float] = (.01, .99),
                trim_type: Literal["clip", "drop"] | None = "clip",
                normalize: Literal["0", "1"] | None = None,
                **kwargs
) -> dict[str, np.ndarray]:
        prop: np.ndarray = np.zeros(data.shape[0])
        
        for fold in np.unique(cvgroup):
                training_fold: np.ndarray = cvgroup != fold
                test_fold: np.ndarray = cvgroup == fold
                prop[test_fold] = method_prop(data, training_fold, test_fold, formula_x, z_column, classification = True, **kwargs)
        
        indices_for_keeping: np.ndarray = np.array([True for _ in range(data.shape[0])])
        
        if trim_type == "drop":
                indices_for_keeping = np.bitwise_and(prop > trim_thresholds[0], prop < trim_thresholds[1])
                prop[np.bitwise_not(indices_for_keeping)] = 0.5
        elif trim_type == "clip":
                prop[prop < trim_thresholds[0]] = trim_thresholds[0]
                prop[prop < trim_thresholds[1]] = trim_thresholds[1]
        
        if normalize == "1":
                mean_value: np.float64 = np.mean(data[indices_for_keeping, z_column] / prop[indices_for_keeping])
                mean_z: np.float64 = np.mean(data[indices_for_keeping])
                prop[indices_for_keeping] = (1 + (1 - mean_z) / (mean_value - mean_z) * (1 - prop[indices_for_keeping]) / prop[indices_for_keeping]) ** (-1)
        elif normalize == "0":
                mean_value: np.float64 = np.mean(1 - data[indices_for_keeping, z_column] / (1 - prop[indices_for_keeping]))
                mean_z: np.float64 = np.mean(data[indices_for_keeping])
                prop[indices_for_keeping] = (1 - (1 + mean_z) / (mean_value - mean_z - 1) * (prop[indices_for_keeping]) / (1 - prop[indices_for_keeping])) ** (-1)
        
        return {"prop": prop, "keep": indices_for_keeping}

def optim_cut_hajek(
                Y: np.ndarray,
                Z: np.ndarray,
                Li: np.ndarray,
                Ui: np.ndarray,
                run_min: bool = False
) -> np.float64:
        if run_min:
                return - optim_cut_hajek(-1 * Y, Z, Li, Ui)
        
        if np.min(Li - Ui) > 0 | np.min(Li) <= 0:
                raise ValueError("Provided lower and upper bounds are not possible")
        
        Y = Y[Z == 1]
        Li = Li[Z == 1]
        Ui = Ui[Z == 1]
        del Z
        sorted_indices: np.ndarray = np.argsort(Y)
        Y = Y[sorted_indices]
        Li = Li[sorted_indices]
        Ui = Ui[sorted_indices]
        del sorted_indices
        start_val: np.float64 = np.mean(Y / Li) / np.mean(1 / Li)
        current_value_index: np.int64 = np.argmax(Y > start_val) + 1
        current_value_numerator: np.float64 = (Y[:current_value_index] / Ui[:current_value_index]).sum() + (Y[current_value_index:] / Li[current_value_index:]).sum()
        current_value_denominator: np.float64 = (1 / Ui[:current_value_index]).sum() + (1 / Li[current_value_index:]).sum()
        current_optimum_value: np.float64 = np.float64(-inf)

        while current_optimum_value < current_value_numerator / current_value_denominator:
                current_optimum_value = current_value_numerator / current_value_denominator
                current_value_numerator = current_value_numerator + (1 / Ui[current_value_index] - 1 / Li[current_value_index]) * Y[current_value_index]
                current_value_denominator = current_value_denominator + (1 / Ui[current_value_index] - 1 / Li[current_value_index])
                current_value_index += 1
        
        return current_optimum_value

def const_regression(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                classification: bool = False,
                **kwargs
) -> np.ndarray:
        y: np.ndarray = get_model_matrix(data, response, form_x)[1]
        model_predictions: np.ndarray = np.array([y[train_mask].mean() for _ in test_mask])
        return model_predictions.reshape(model_predictions.shape[0])

def boost_regression(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                classification: bool = False,
                **kwargs
) -> np.ndarray:
        X, y = get_model_matrix(data, response, form_x)
        
        if classification:
                classifier: GradientBoostingClassifier = GradientBoostingClassifier(**kwargs)
                classifier.fit(X[train_mask, :], np.int32(y[train_mask]))
                model_predictions: np.ndarray = classifier.predict_proba(X[test_mask, :])[:, 1]
                return model_predictions.reshape(model_predictions.shape[0])
        
        
        regressor: GradientBoostingRegressor = GradientBoostingRegressor(**kwargs)
        regressor.fit(X[train_mask, :], y[train_mask])
        model_predictions: np.ndarray = regressor.predict(X[test_mask, :])
        # TODO: procedure for optimal number of regressors
        return model_predictions.reshape(model_predictions.shape[0])

def svm_regression(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                classification: bool = False,
                **kwargs
) -> np.ndarray:
        X, y = get_model_matrix(data, response, form_x)

        if classification:
                classifier: SVC = SVC(gamma = "auto", **kwargs)
                classifier.fit(X[train_mask, :], np.int32(y[train_mask]))
                model_predictions: np.ndarray = classifier.predict_proba(X[test_mask, :])[:, 1]
                return model_predictions.reshape(model_predictions.shape[0])
        
        regressor: SVR = SVR(gamma = "auto", **kwargs)
        regressor.fit(X[train_mask, :], y[train_mask])
        model_predictions: np.ndarray = regressor.predict(X[test_mask])
        return model_predictions.reshape(model_predictions.shape[0])

def forest_regression(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                classification: bool = False,
                **kwargs
) -> np.ndarray:
        X, y = get_model_matrix(data, response, form_x)
        
        if classification:
                classifier: RandomForestClassifier = RandomForestClassifier(**kwargs)
                classifier.fit(X[train_mask, :], np.int32(y[train_mask]))
                model_predictions: np.ndarray = classifier.predict_proba(X[test_mask, :])[:, 1]
                return model_predictions.reshape(model_predictions.shape[0])
        
        regressor: RandomForestRegressor = RandomForestRegressor(**kwargs)
        regressor.fit(X[train_mask, :], y[train_mask])
        model_predictions: np.ndarray = regressor.predict(X[test_mask, :])
        return model_predictions.reshape(model_predictions.shape[0])

def linear_regression(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                classification: bool = False,
                **kwargs
) -> np.ndarray:
        X, y = get_model_matrix(data, response, form_x)

        if classification:
                classifier: LogisticRegression = LogisticRegression(
                        penalty=None,
                        **kwargs
                )
                classifier.fit(X[train_mask, :], np.int32(y[train_mask]))
                model_predictions: np.ndarray = classifier.predict_proba(X[test_mask, :])[:, 1]
                return model_predictions.reshape(model_predictions.shape[0])
        
        regressor: LinearRegression = LinearRegression(**kwargs)
        regressor.fit(X[train_mask, :], y[train_mask])
        model_predictions: np.ndarray = regressor.predict(X[test_mask, :])
        return model_predictions.reshape(model_predictions.shape[0])

def const_quantile(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                tau: float,
                **kwargs
) -> np.ndarray:
        y: np.ndarray = get_model_matrix(data, response, form_x)[1]
        model_predictions: np.ndarray = np.array([np.quantile(y, tau) for _ in test_mask])
        return model_predictions.reshape(model_predictions.shape[0])

def linear_quantile(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                tau: float,
                **kwargs
) -> np.ndarray:
        X, y = get_model_matrix(data, response, form_x)
        
        try:
                quantile_regressor: QuantileRegressor = QuantileRegressor(
                        quantile = tau,
                        **kwargs
                )
                quantile_regressor.fit(X[train_mask, :], y[train_mask])
                model_predictions: np.ndarray = quantile_regressor.predict(X[test_mask, :])
                return model_predictions.reshape(model_predictions.shape[0])
        except Exception as e:
                warnings.warn(f"Warning: {e}. Failed to fit conditional model; reverting to marginal model")
                return const_quantile(
                        data,
                        train_mask,
                        test_mask,
                        form_x,
                        response,
                        tau,
                        **kwargs
                )

def forest_quantile(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                tau: float,
                **kwargs
) -> np.ndarray:
        X, y = get_model_matrix(data, response, form_x)
        
        if "pretrained_forest" in kwargs:
                try:
                        model_predictions: np.ndarray = kwargs["pretrained_forest"].predict(X[test_mask, :], quantiles = tau)
                        return model_predictions.reshape(model_predictions.shape[0])
                except Exception as e:
                        warnings.warn(f"Warning: {e}. Failed to fit pre-trained forest model; reverting to re-trained forest model")
                        return forest_quantile(
                                data,
                                train_mask,
                                test_mask,
                                form_x,
                                response,
                                tau
                        )
        
        try:
                quantile_regressor: RandomForestQuantileRegressor = RandomForestQuantileRegressor(
                        default_quantiles = tau,
                        **kwargs
                )
                quantile_regressor.fit(X[train_mask, :], y[train_mask])
                model_predictions: np.ndarray = quantile_regressor.predict(X[test_mask, :])
                return model_predictions.reshape(model_predictions.shape[0])
        except Exception as e:
                warnings.warn(f"Warning: {e}. Failed to fit forest model; reverting to conditional model")
                return linear_quantile(
                        data,
                        train_mask,
                        test_mask,
                        form_x,
                        response,
                        tau
                )

def forest_conditional_dist(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                classification: bool = False,
                **kwargs
) -> np.ndarray:
        X, y = get_model_matrix(data, response, form_x)
        
        if classification:
                classifier: RandomForestClassifier = RandomForestClassifier(**kwargs)
                classifier.fit(X[train_mask, :], np.int32(y[test_mask]))
                train_terminal_nodes: np.ndarray = np.array(
                        classifier.apply(X[train_mask, :])
                )
                test_terminal_nodes: np.ndarray = np.array(
                        classifier.apply(X[test_mask, :])
                )
        else:
                regressor: RandomForestRegressor = RandomForestRegressor(**kwargs)
                regressor.fit(X[train_mask, :], y[train_mask])
                train_terminal_nodes: np.ndarray = np.array(
                        regressor.apply(X[train_mask, :])
                )
                test_terminal_nodes: np.ndarray = np.array(
                        regressor.apply(X[test_mask, :])
                )
        
        w: np.ndarray = np.zeros((test_mask.shape[0], train_mask.shape[0]))
        
        for index, row in enumerate(w):
                P: np.ndarray = np.equal(train_terminal_nodes, test_terminal_nodes[index])
                w[index] = ((P / P.sum(axis = 0)).sum(axis = 1) / P.shape[1]).T
        
        return w

def kernel_conditional_dist(
                data: pd.DataFrame,
                train_mask: np.ndarray,
                test_mask: np.ndarray,
                form_x: str,
                response: str,
                classification: bool = False,
                **kwargs
) -> np.ndarray:
        
        if "h" not in kwargs:
                h: float = .1
        else:
                h = kwargs["h"]
        
        X: np.ndarray = get_model_matrix(data, response, form_x)[0]
        X = X[:, 1:]
        X_varcov: np.ndarray = np.cov(X, rowvar = False)
        varcov_decomp: np.ndarray = np.linalg.cholesky(X_varcov)
        tmp: np.ndarray = np.linalg.solve(varcov_decomp, X.T)
        distance2: np.ndarray = euclidean_distances(tmp.T, tmp.T) ** 2
        w: np.ndarray = 1. * (distance2[test_mask, train_mask] <= h)
        return w

def binary_extrapolation_quantile(
                tau: float,
                muhat: np.ndarray
) -> np.ndarray:
        return (muhat >= (1 - tau)).astype(np.int32)

def binary_extrapolation_kappa(
                tau: float,
                muhat: np.ndarray,
                **kwargs

) -> np.ndarray:
        Qhat = binary_extrapolation_quantile(
                        tau,
                        muhat
                )
        Lambda = np.max([tau / (1 - tau), (1 - tau) / tau])
        lambda_exp_sign: np.int32 = np.sign(tau - 0.5)
        kappa_first_term: np.ndarray = Qhat * (1 - Lambda ** (-lambda_exp_sign)) * (1 - muhat)
        kappa_second_term: np.ndarray = (1 - Qhat) * Lambda ** (lambda_exp_sign) * muhat
        kappa: np.ndarray = kappa_first_term + kappa_second_term
        return kappa

def bootstrap_props_iteration(
                data: pd.DataFrame,
                form_x: str,
                form_z: str,
                K: int,
                method_prop: Callable,
                iteration_num: int,
                refit_propensities: bool,
                trim_thresholds: tuple[float, float],
                trim_type: Literal["clip", "drop"] | None,
                normalize: Literal["0", "1"] | None,
                prop: dict[str, np.ndarray]

):
        np.random.seed(iteration_num) # TODO: change to best practice
        cvboot = make_cvgroup_balanced(data, K, form_z)
        sample_indices: np.ndarray = np.random.default_rng().choice(
                data.shape[0], 
                size = data.shape[0]
        )
        if refit_propensities:
                propensities: dict[str, np.ndarray] = cross_fit_propensities(
                        data.iloc[sample_indices, :],
                        cvboot[sample_indices],
                        form_x,
                        form_z,
                        method_prop,
                        trim_thresholds = trim_thresholds,
                        trim_type = trim_type,
                        normalize = normalize
                )
                cv_propensities: np.ndarray = propensities["prop"]
        else:
                cv_propensities: np.ndarray = prop["prop"]
        iteration_results: pd.DataFrame = pd.DataFrame({
                "original_indices": sample_indices,
                "boot_indices": np.arange(data.shape[0]),
                "prop": cv_propensities
        })
        cv_groups_data: pd.DataFrame = pd.DataFrame({
                "original_indices": np.arange(data.shape[0]),
                "cv": cvboot
        })
        return pd.merge(
                iteration_results, 
                cv_groups_data, 
                how = "left",
                on = "original_indices"
        )

def bootstrap_props(
                data: pd.DataFrame,
                form_x: str,
                form_z: str,
                K: int,
                method_prop: Callable,
                trim_thresholds: tuple[float, float],
                trim_type: Literal["clip", "drop"] | None,
                normalize: Union[Literal["0", "1"], None],
                prop: dict[str, np.ndarray],
                bootstrap_num: int,
                reset_seed: bool,
                refit_propensities: bool,
                **kwargs
) -> list[pd.DataFrame]:
        start_boot_seed: dict[str, Any] = np.random.get_state()
        boot_data = [
                bootstrap_props_iteration(
                        data,
                        form_x,
                        form_z,
                        K,
                        method_prop,
                        iteration_num,
                        refit_propensities,
                        trim_thresholds,
                        trim_type,
                        normalize,
                        prop,
                        **kwargs
                )
                for iteration_num in range(bootstrap_num)
        ]
        if reset_seed:
                np.random.set_state(start_boot_seed)
        
        return boot_data

def summarize_boots(
                boot_estimates: dict[str, pd.DataFrame]
) -> tuple[
        dict[str, np.float64 | float],
        dict[str, np.float64 | float], 
        dict[str, np.float64 | float], 
        dict[str, np.float64 | float], 
        dict[str, np.float64 | float]
]:
        
        # TODO all nan when no boot_estimates
        ci_mean: dict[str, np.float64 | float] = {}
        ci_50: dict[str, np.float64 | float] = {}
        ci_90: dict[str, np.float64 | float] = {}
        ci_95: dict[str, np.float64 | float] = {}
        ci_sd: dict[str, np.float64 | float] = {}

        if len(boot_estimates) < 4:
                for t in [0, 1]:
                        for z in [0, 1]:
                                ci_mean[f"{z}{t}"] = np.nan
                                ci_50[f"{z}{t}"] = np.nan
                                ci_90[f"{z}{t}"] = np.nan
                                ci_95[f"{z}{t}"] = np.nan
                                ci_sd[f"{z}{t}"] = np.nan
                                ci_mean[f"AT{z}{t}"] = np.nan
                                ci_50[f"AT{z}{t}"] = np.nan
                                ci_90[f"AT{z}{t}"] = np.nan
                                ci_95[f"AT{z}{t}"] = np.nan
                                ci_sd[f"AT{z}{t}"] = np.nan
                
                ci_mean["1100"] = np.nan
                ci_50["1100"] = np.nan
                ci_90["1100"] = np.nan
                ci_95["1100"] = np.nan
                ci_sd["1100"] = np.nan
                ci_mean["1001"] = np.nan
                ci_50["1001"] = np.nan
                ci_90["1001"] = np.nan
                ci_95["1001"] = np.nan
                ci_sd["1001"] = np.nan
                return ci_mean, ci_50, ci_90, ci_95, ci_sd

        for t in [0, 1]:
                for z in [0, 1]:
                        ci_mean[f"{z}{t}"] = np.mean(boot_estimates[f"{z}{t}"]["if"])
                        ci_50[f"{z}{t}"] = np.quantile(boot_estimates[f"{z}{t}"]["if"], .5)
                        ci_90[f"{z}{t}"] = np.quantile(boot_estimates[f"{z}{t}"]["if"], t * .95 + (1 - t) * .05)
                        ci_95[f"{z}{t}"] = np.quantile(boot_estimates[f"{z}{t}"]["if"], t * .975 + (1 - t) * .025)
                        ci_sd[f"{z}{t}"] = np.std(boot_estimates[f"{z}{t}"]["if"], ddof = 1)
                        att_atc_bounds: np.ndarray = ((boot_estimates[f"{z}1"]["Y_mean"] - boot_estimates[f"{1 - z}{1 - t}"]["if"]) / boot_estimates[f"{z}1"]["Z_mean"]).to_numpy()
                        ci_mean[f"AT{z}{t}"] = np.mean(att_atc_bounds)
                        ci_50[f"AT{z}{t}"] = np.quantile(att_atc_bounds, .5)
                        ci_90[f"AT{z}{t}"] = np.quantile(att_atc_bounds, t * .95 + (1 - t) * .05)
                        ci_95[f"AT{z}{t}"] = np.quantile(att_atc_bounds, t * .975 + (1 - t) * .025)
                        ci_sd[f"AT{z}{t}"] = np.std(boot_estimates[f"{z}{t}"]["if"], ddof = 1)
        
        ate_1100_bounds: np.ndarray = (boot_estimates["11"]["if"] - boot_estimates["00"]["if"]).to_numpy()
        ci_mean["1100"] = np.mean(ate_1100_bounds)
        ci_50["1100"] = np.quantile(ate_1100_bounds, .5)
        ci_90["1100"] = np.quantile(ate_1100_bounds, 0.95)
        ci_95["1100"] = np.quantile(ate_1100_bounds, 0.975)
        ci_sd["1100"] = np.std(ate_1100_bounds, ddof = 1)
        ate_1001_bounds: np.ndarray = (boot_estimates["10"]["if"] - boot_estimates["01"]["if"]).to_numpy()
        ci_mean["1001"] = np.mean(ate_1001_bounds)
        ci_50["1001"] = np.quantile(ate_1001_bounds, .5)
        ci_90["1001"] = np.quantile(ate_1001_bounds, 0.95)
        ci_95["1001"] = np.quantile(ate_1001_bounds, 0.975)
        ci_sd["1001"] = np.std(ate_1001_bounds, ddof = 1)
        return ci_mean, ci_50, ci_90, ci_95, ci_sd
                        
def summarize_results(
                lambda_value: float | np.float64,
                data: pd.DataFrame,
                y_column: str,
                z_column: str,
                influence_function: dict[str, np.ndarray],
                se_results: dict[str, np.float64],
                boot_facts: tuple
) -> pd.DataFrame:
        ci_mean, ci_50, ci_90, ci_95, ci_sd = boot_facts
        att_plus: np.float64 = np.mean(data[y_column] - influence_function["00"]) / np.mean(data[z_column])
        att_minus: np.float64 = np.mean(data[y_column] - influence_function["01"]) / np.mean(data[z_column])
        atc_plus: np.float64 = np.mean(data[y_column] - influence_function["10"]) / np.mean(1 - data[z_column])
        atc_minus: np.float64 = np.mean(data[y_column] - influence_function["11"]) / np.mean(1 - data[z_column])
        summary: pd.DataFrame = pd.DataFrame({
                "Lambda": [lambda_value for _ in range(10)],
                "estimand": ["1", "1", "0", "0", "ATE", "ATE", "ATT", "ATT", "ATC", "ATC"],
                "side": ["upper", "lower"] * 5,
                "estimate": [
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
                ],
                "sterr_if": [
                        se_results["11"],
                        se_results["10"],
                        se_results["01"],
                        se_results["00"],
                        se_results["1100"],
                        se_results["1001"],
                        np.std((data[y_column].to_numpy() - influence_function["00"] - data[z_column].to_numpy() * att_plus) / np.mean(data[z_column].to_numpy())) / np.sqrt(data.shape[0]),
                        np.std((data[y_column].to_numpy() - influence_function["01"] - data[z_column].to_numpy() * att_plus) / np.mean(data[z_column].to_numpy())) / np.sqrt(data.shape[0]),
                        np.std((data[y_column].to_numpy() - influence_function["10"] - (1 - data[z_column].to_numpy()) * atc_plus) / np.mean(data[z_column].to_numpy())) / np.sqrt(data.shape[0]),
                        np.std((data[y_column].to_numpy() - influence_function["11"] - (1 - data[z_column].to_numpy()) * atc_plus) / np.mean(data[z_column].to_numpy())) / np.sqrt(data.shape[0])
                ],
                "sterr_boot": [
                        ci_sd["11"],
                        ci_sd["10"],
                        ci_sd["01"],
                        ci_sd["00"],
                        ci_sd["1100"],
                        ci_sd["1001"],
                        ci_sd["AT11"],
                        ci_sd["AT10"],
                        ci_sd["AT01"],
                        ci_sd["AT00"]
                ],
                "bootmean": [
                        ci_mean["11"],
                        ci_mean["10"],
                        ci_mean["01"],
                        ci_mean["00"],
                        ci_mean["1100"],
                        ci_mean["1001"],
                        ci_mean["AT11"],
                        ci_mean["AT10"],
                        ci_mean["AT01"],
                        ci_mean["AT00"]
                ],
                "quantiles50": [
                        ci_50["11"],
                        ci_50["10"],
                        ci_50["01"],
                        ci_50["00"],
                        ci_50["1100"],
                        ci_50["1001"],
                        ci_50["AT11"],
                        ci_50["AT10"],
                        ci_50["AT01"],
                        ci_50["AT00"]
                ],
                "quantiles90": [
                        ci_90["11"],
                        ci_90["10"],
                        ci_90["01"],
                        ci_90["00"],
                        ci_90["1100"],
                        ci_90["1001"],
                        ci_90["AT11"],
                        ci_90["AT10"],
                        ci_90["AT01"],
                        ci_90["AT00"]
                ],
                "quantiles95": [
                        ci_95["11"],
                        ci_95["10"],
                        ci_95["01"],
                        ci_95["00"],
                        ci_95["1100"],
                        ci_95["1001"],
                        ci_95["AT11"],
                        ci_95["AT10"],
                        ci_95["AT01"],
                        ci_95["AT00"]
                ],
        })
        return summary

def dvds(
                Lambdas: list[float] | NDArray[np.float64],
                data: pd.DataFrame,
                form_x: str,
                z_column: str,
                y_column: str,
                method_prop: Callable,
                options_prop: dict[str, Any],
                method_quant: Callable = const_quantile,
                options_quant: dict[str, Any] = {},
                method_regn: Callable = const_regression,
                options_regn: dict[str, Any] = {},
                method_conddist: Callable = kernel_conditional_dist,
                options_conddist: dict[str, Any] = {},
                conddist_quant: bool = False,
                conddist_kappa: bool = False,
                K: int = 5,
                semiadaptive: bool = False,
                trim_thresholds: tuple[float, float] = (0.01, 0.99),
                trim_type: Literal["clip", "drop"] | None = "clip",
                normalize: Literal["0", "1"] | None = None,
                form_x_quant: str | None = None,
                form_x_kappa: str | None = None,
                stabilize_obj: bool = False,
                boot_infer: bool = False,
                boot_settings: dict[str, Any] = {
                        "refit_propensities": False,
                        "bootstrap_num": 500,
                        "reset_seed": True
                }
) -> pd.DataFrame:
        
        classification: bool = pd.unique(data[y_column]).shape[0] == 2
        cvgroup: np.ndarray = make_cvgroup_balanced(data, K, z_column)
        Lambdas = np.array(Lambdas)

        if form_x_quant is None:
                form_x_quant = form_x
        
        if form_x_kappa is None:
                form_x_kappa = form_x
        
        propensities_result: dict[str, np.ndarray] = cross_fit_propensities(
                data, 
                cvgroup,
                form_x,
                z_column,
                method_prop,
                trim_thresholds = trim_thresholds,
                trim_type = trim_type,
                normalize = normalize,
                **options_prop
        )

        data = data[propensities_result["keep"]]
        cvgroup = cvgroup[propensities_result["keep"]]
        propensities_hat: np.ndarray = propensities_result["prop"][propensities_result["keep"]]

        if (method_regn == binary_extrapolation_kappa) | (method_quant == binary_extrapolation_quantile):
                                
                if "muhat_function" not in options_quant: raise ArgumentError(
                        "`muhat_function` must be in in options_quant \
                        and must be a function for estimating binary \
                        classification when binary extrapolation is \
                        used"
                )

                mu_hats: dict[str, np.ndarray] = {}
                
                for z in [0, 1]:
                        mu_hats[f"{z}"] = np.zeros((data.shape[0],))
                        for fold in np.unique(cvgroup):
                                train_mask: np.ndarray = cvgroup != fold
                                
                                if not semiadaptive:
                                        odd_folds_mask: np.ndarray = (cvgroup - (cvgroup > fold)) % 2 == 1
                                        train_mask = np.logical_and(
                                                train_mask,
                                                odd_folds_mask
                                        )
                                
                                train_mask = np.logical_and(
                                        train_mask,
                                        data[z_column] == z
                                )
                                mu_hats[f"{z}"][cvgroup == fold] = options_quant["muhat_function"](
                                        data,
                                        train_mask,
                                        cvgroup == fold,
                                        form_x_quant,
                                        y_column,
                                        classification = classification,
                                        **options_quant
                                )
        
        if boot_infer:
                boot_data: list[pd.DataFrame] = bootstrap_props(
                        data,
                        form_x,
                        z_column,
                        K,
                        method_prop,
                        trim_thresholds,
                        trim_type,
                        normalize,
                        propensities_result,
                        **boot_settings
                )
        
        if conddist_kappa | conddist_quant:

                w: dict[str, np.ndarray] = {}
                wc: dict[str, Any] = {}

                for z in [0, 1]:

                        w[f"{z}"] = np.zeros((data.shape[0], data.shape[0]))

                        for fold in np.unique(cvgroup):
                                train_mask: np.ndarray = cvgroup != fold
                                train_mask = np.logical_and(
                                        train_mask,
                                        data[z_column] == z
                                )
                                w[f"{z}"][cvgroup == fold, train_mask] = method_conddist(
                                        data,
                                        train_mask,
                                        cvgroup == fold,
                                        form_x,
                                        y_column,
                                        classification,
                                        **options_conddist
                                )
                        
                        w[f"{z}"] = w[f"{z}"] / w[f"{z}"].sum(1)
                        wc[f"{z}"] = w[f"{z}"].cumsum(1)
                        
        if (
                (method_quant == forest_quantile)
                and ("reuse_forests" in options_quant)
                and (options_quant["reuse_forests"])
        ):
                pretrained_forests: dict[str, RandomForestQuantileRegressor] = {}
                all_taus: np.ndarray = np.hstack([
                        1 / (1 + Lambdas),
                        Lambdas / (1 + Lambdas)
                ])
                all_taus = np.unique(all_taus)
                all_taus.sort()
                X, y = get_model_matrix(data, y_column, form_x_quant)
                X = X[:, 1:]

                for fold in np.unique(cvgroup):
                        for z in [0, 1]:
                                train_mask: np.ndarray = cvgroup != fold
                                train_mask = np.logical_and(
                                        train_mask,
                                        data[z_column] == z
                                )
                                pretrained_forests[f"{fold}{z}"] = RandomForestQuantileRegressor(
                                        default_quantiles = all_taus.tolist(),
                                        **options_quant
                                ).fit(
                                        X[train_mask],
                                        y[train_mask]
                                )
                
                options_quant["pretrained_forests"] = pretrained_forests
        
        final_results: pd.DataFrame = pd.DataFrame()

        for lambda_value in tqdm(Lambdas):
                se_results: dict[str, np.float64] = {}
                efficient_influence_function: dict[str, np.ndarray] = {}
                boot_estimates: dict[str, pd.DataFrame] = {}

                for z in [0, 1]:
                        for t in [0, 1]:
                                q: np.ndarray = np.zeros((data.shape[0],))
                                tau = (1 - t + t * lambda_value) / (lambda_value + 1)

                                if conddist_quant:
                                        q = data[y_column][np.argmax(wc[f"{z}"], 1)].to_numpy()
                                elif method_quant == binary_extrapolation_quantile:
                                        q = method_quant(tau, mu_hats[f"{z}"]) # type: ignore
                                else:
                                        for fold in np.unique(cvgroup):
                                                try:
                                                        options_quant["pretrained_forest"] = options_quant["pretrained_forests"][f"{fold}{z}"]
                                                except KeyError:
                                                        pass
                                                
                                                train_mask: np.ndarray = cvgroup != fold

                                                if not semiadaptive:
                                                        even_folds_mask: np.ndarray = (cvgroup - (cvgroup > fold)) % 2 == 0
                                                        train_mask = np.logical_and(
                                                                train_mask,
                                                                even_folds_mask
                                                        )

                                                train_mask = np.logical_and(
                                                        train_mask,
                                                        data[z_column] == z
                                                )
                                                q[cvgroup == fold] = method_quant(
                                                        data,
                                                        train_mask,
                                                        cvgroup == fold,
                                                        form_x,
                                                        y_column,
                                                        tau,
                                                        **options_quant
                                                )
                                
                                kappa: np.ndarray = np.zeros((data.shape[0],))
                                data["_hinge"] = q + lambda_value ** (np.sign((2 * t - 1) * (data[y_column] - q))) * (data[y_column] - q)

                                if conddist_kappa:
                                        kappa = w[f"{z}"] @ data["_hinge"].to_numpy()
                                elif method_regn == binary_extrapolation_kappa:
                                        kappa = method_regn(tau, mu_hats[f"{z}"])
                                else:
                                        for fold in np.unique(cvgroup):
                                                train_mask: np.ndarray = cvgroup != fold
                                
                                                if not semiadaptive:
                                                        even_folds_mask: np.ndarray = (cvgroup - (cvgroup > fold)) % 2 == 0
                                                        train_mask = np.logical_and(
                                                                train_mask,
                                                                even_folds_mask
                                                        )
                                                
                                                train_mask = np.logical_and(
                                                        train_mask,
                                                        data[z_column] == z
                                                )
                                                kappa[cvgroup == fold] = method_regn(
                                                        data,
                                                        train_mask,
                                                        cvgroup == fold,
                                                        form_x,
                                                        "_hinge",
                                                        **options_regn
                                                )
                                
                                zz: np.ndarray = (2 * z - 1) * data[z_column].to_numpy() + (1 - z)
                                ee: np.ndarray = (2 * z - 1) * propensities_hat + (1 - z)

                                if stabilize_obj:
                                        ws = zz * ((1 / ee) / np.mean(zz / ee) - 1)
                                        efficient_influence_function[f"{z}{t}"] = zz * data[y_column].to_numpy() + (1 - zz) * kappa + ws * (data["_hinge"].to_numpy() - kappa)
                                else:
                                        efficient_influence_function[f"{z}{t}"] = kappa + zz * (q - kappa) / ee + zz * (data[y_column].to_numpy() - q) * (ee + (1 - ee) * lambda_value ** ((2 * t - 1) * np.sign(data[y_column].to_numpy() - q))) / ee
                                
                                if boot_infer:
                                        start_boot_seed: dict[str, Any] = np.random.get_state()
                                        boot_estimates[f"{z}{t}"] = pd.DataFrame()

                                        for iteration in range(boot_settings["bootstrap_num"]):
                                                np.random.seed(iteration)
                                                index_star: np.ndarray = boot_data[iteration]["original_indices"].to_numpy()
                                                data_star: pd.DataFrame = data[index_star].copy()
                                                data_star["prop"] = boot_data[iteration]["prop"]
                                                q_star: np.ndarray = q[index_star]
                                                kappa_star: np.ndarray = kappa[index_star]
                                                propensities_hat_star: np.ndarray = propensities_hat[index_star]
                                                Y_star: np.ndarray = data_star[y_column].to_numpy()
                                                zz_star: np.ndarray = (2 * z - 1) * data_star[z_column].to_numpy() + 1 - z
                                                ee_star = (2 * z - 1) * propensities_hat_star + 1 - z

                                                if stabilize_obj:
                                                        ws_star: np.ndarray = zz_star * ((1 / ee_star) / np.mean(zz_star / ee_star) - 1)
                                                        hinge_star: np.ndarray = q_star + lambda_value ** ((2 * t - 1) * np.sign(Y_star - q_star)) * (Y_star * q_star)
                                                        mean_eif_star: np.float64 = np.mean(zz_star * Y_star + + (1 - zz_star) * kappa_star + ws_star * (hinge_star - kappa_star))
                                                else:
                                                        mean_eif_star: np.float64 = np.mean(kappa_star + zz_star * (q_star - kappa_star) / ee_star + zz_star * (Y_star - q_star) * (ee_star + (1 - ee_star) * lambda_value ** ((2 * t - 1) * np.sign(Y_star - q))) / ee_star)
                                                
                                                boot_iteration_row: pd.DataFrame = pd.DataFrame({
                                                        "if": [mean_eif_star],
                                                        "Y_mean": [Y_star.mean()],
                                                        "Z_mean": [zz_star.mean()]
                                                })
                                                boot_estimates[f"{z}{t}"] = pd.concat([boot_estimates[f"{z}{t}"], boot_iteration_row])
                                        
                                        if boot_settings["reset_seed"]:
                                                np.random.set_state(start_boot_seed)
                                
                                se_results[f"{z}{t}"] = np.std(efficient_influence_function[f"{z}{t}"], ddof = 1) / np.sqrt(data.shape[0])
                
                se_results["1100"] = np.std(efficient_influence_function["11"] - efficient_influence_function["00"], ddof = 1) / np.sqrt(data.shape[0])
                se_results["1001"] = np.std(efficient_influence_function["10"] - efficient_influence_function["01"], ddof = 1) / np.sqrt(data.shape[0])
                boot_facts: tuple = summarize_boots(boot_estimates)
                lambda_results: pd.DataFrame = summarize_results(
                        lambda_value,
                        data,
                        y_column,
                        z_column,
                        efficient_influence_function,
                        se_results,
                        boot_facts
                )
                final_results = pd.concat([final_results, lambda_results])
        
        return final_results