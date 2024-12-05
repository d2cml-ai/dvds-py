import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import norm
from scipy.special import logit
from tqdm import tqdm
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
# from treeinterpreter import treeinterpreter as ti

# Helper functions ##########################

def phi_continuous(X: np.ndarray, Y: np.array, Z: np.array, mu: np.array, e: np.array, Lambdas: np.array): # VAL: Cambié Y y Z matrices a arrays
    # DVDS generic influence function for continuous outcomes
    n = len(Y)
    m = len(Lambdas)
    taus = Lambdas / (Lambdas + 1)

    # Estimate quantiles using quantile forests
    Q = np.zeros((n, m))
    qfit = RandomForestQuantileRegressor().fit(X[Z==1], Y[Z==1])
    Q[Z==1] = qfit.predict(X[Z==1], Y, quantiles=taus)
    Q[Z==0] = qfit.predict(X[Z==0], Y, quantiles=taus)

    # Estimate CV@R using random forests. Uses the GRF weights from the median tau for all taus
    cvar = np.zeros((n, m))
    k = int(np.median(range(0, m)))
    check = np.array([Q[:, i] + np.maximum(0, Y - Q[:, i]) / (1 - taus[i]) for i in range(m)]).T
    cvarfit = RandomForestRegressor().fit(X[Z==1], check[Z==1, k-1])
    # _, _, contributions = ti.predict(cvarfit, X[Z==1]) ### creo que va por acá
    # cvar[Z==1] = # FALTA
    # cvar[Z==0] = # FALTA

    # Form the influence function
    eif = np.array([Z * Y + (1 - Z) * ((1 / Lambdas[i]) * mu + (1 - 1 / Lambdas[i]) * cvar[:, i]) +
                   (1 - e) / e * Z * (((1 / Lambdas[i]) * Y + (1 - 1 / Lambdas[i]) * check[:, i]) -
                                      ((1 / Lambdas[i]) * mu + (1 - 1 / Lambdas[i]) * cvar[:, i]))
                   for i in range(m)]).T

    return eif


# Main DVDS functions ##########################

# X is only used for continuous, but included here to analogize 
def dvds_binary(Y: np.array, Z: np.array, mu0, mu1, e, Lambda=1, alpha=0.05):
    # DVDS sensitivity analysis for binary outcomes
    tau = Lambda / (Lambda + 1)
    Q1plus = mu1 > 1 - tau
    Q0plus = mu0 > 1 - tau
    Q1mins = mu1 > tau
    Q0mins = mu0 > tau
    kappa1plus = np.minimum(1 - 1 / Lambda + mu1 / Lambda, mu1 * Lambda)
    kappa0plus = np.minimum(1 - 1 / Lambda + mu0 / Lambda, mu0 * Lambda)
    kappa1mins = np.maximum(1 - Lambda + mu1 * Lambda, mu1 / Lambda)
    kappa0mins = np.maximum(1 - Lambda + mu0 * Lambda, mu0 / Lambda)
    ATEplus1 = Z * Y + (1 - Z) * kappa1plus + (1 - e) / e * Z * (Q1plus + Lambda ** np.sign(Y - Q1plus) * (Y - Q1plus) - kappa1plus)
    ATEplus0 = (1 - Z) * Y + Z * kappa0plus + e / (1 - e) * (1 - Z) * (Q0plus + Lambda ** np.sign(Y - Q0plus) * (Y - Q0plus) - kappa0plus)
    ATEmins1 = Z * Y + (1 - Z) * kappa1mins + (1 - e) / e * Z * (Q1mins + Lambda ** np.sign(Q1mins - Y) * (Y - Q1mins) - kappa1mins)
    ATEmins0 = (1 - Z) * Y + Z * kappa0mins + e / (1 - e) * (1 - Z) * (Q0mins + Lambda ** np.sign(Q0mins - Y) * (Y - Q0mins) - kappa0mins)
    ATEplus =  ATEplus1 - ATEmins0
    ATEmins = ATEmins1 - ATEplus0
    c = norm.ppf(1 - alpha / 2) / np.sqrt(len(Y))
    
    results = {
            "Lambda": Lambda,
            "upper": np.mean(ATEplus),
            "lower": np.mean(ATEmins),
            "upper.CI": np.mean(ATEplus) + c * np.std(ATEplus, ddof=1),
            "lower.CI": np.mean(ATEmins) - c * np.std(ATEmins, ddof=1),
            "upper.1": np.mean(ATEplus1),
            "lower.1": np.mean(ATEmins1),
            "upper.0": np.mean(ATEplus0), # VALE: En el source code ponen ATEmin0
            "lower.0": np.mean(ATEmins0) # VALE: En el source code ponen ATEplus0
    }
    return pd.DataFrame(results)

def dvds_continuous(X: np.ndarray, Y: np.array, Z: np.array, mu0, mu1, e, Lambdas=1, alpha=0.05):
    # DVDS sensitivity analysis for continuous outcomes
    # Uses random forests for all quantile and CV@R nuisances
    n = len(Y)
    phi1plus = phi_continuous(X, Y, Z, mu1, e, Lambdas)
    phi1mins = -phi_continuous(X, -Y, Z, -mu1, e, Lambdas)
    phi0plus = phi_continuous(X, Y, 1-Z, mu0, 1-e, Lambdas)
    phi0mins = -phi_continuous(X, -Y, 1-Z, -mu0, 1-e, Lambdas)
    ATEplus = phi1plus - phi0mins
    ATEmins = phi1mins - phi0plus
    c = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    
    results = []
    for i in range(len(Lambdas)):
        result = {
            "Lambda": Lambdas[i],
            "lower": np.mean(ATEmins[:, i]),
            "upper": np.mean(ATEplus[:, i]),
            "lower.CI": np.mean(ATEmins[:, i]) - c * np.std(ATEmins[:, i]),
            "upper.CI": np.mean(ATEplus[:, i]) + c * np.std(ATEplus[:, i]),
            "upper.1": np.mean(phi1plus),
            "lower.1": np.mean(phi1mins),
            "upper.0": np.mean(phi0plus),
            "lower.0": np.mean(phi0mins)
        }
        results.append(result)
    
    return pd.DataFrame(results)


# ZSB comparison ##########################

def extrema(A, Y: np.array, gamma, fitted_prob):
    # Helper function for ZSB sensitivity analysis
    fitted_logit = logit(fitted_prob)
    eg = np.exp(-fitted_logit)
    Y = Y[A == 1]
    eg = eg[A == 1]
    Y = Y[np.argsort(-Y)]
    eg = eg[np.argsort(-Y)]
    num_each_low = Y * (1 + np.exp(-gamma) * eg)
    num_each_up = Y * (1 + np.exp(gamma) * eg)
    num = np.concatenate(([0], np.cumsum(num_each_up))) + np.concatenate((np.cumsum(num_each_low[::-1])[::-1], [0]))
    den_each_low = (1 + np.exp(-gamma) * eg)
    den_each_up = (1 + np.exp(gamma) * eg)
    den = np.concatenate(([0], np.cumsum(den_each_up))) + np.concatenate((np.cumsum(den_each_low[::-1])[::-1], [0]))
    maximum = np.max(num / den)
    num = np.concatenate(([0], np.cumsum(num_each_low))) + np.concatenate((np.cumsum(num_each_up[::-1])[::-1], [0]))
    den = np.concatenate(([0], np.cumsum(den_each_low))) + np.concatenate((np.cumsum(den_each_up[::-1])[::-1], [0]))
    minimum = np.min(num / den)

    return minimum, maximum

def extrema_os(Z: np.arra, Y: np.arra, Lambda: np.arra, e: np.arra):
    return extrema(Z, Y, np.log(Lambda), e) - np.flip(extrema(1 - Z, Y, np.log(Lambda), 1 - e))

def extrema_aipw(Z: np.arra, Y: np.arra, Lambda: np.arra, e: np.arra, mu0: np.arra, mu1: np.arra):
    # ZSB sensitivity analysis for AIPW
    eps = Y - np.where(Z == 1, mu1, mu0)
    bounds = np.mean(mu1 - mu0) + extrema_os(Z, eps, Lambda, e)
    
    result = {
        "Lambda": [Lambda] * 2,
        "upper": [bounds[1]],
        "lower": [bounds[0]],
        "upper.CI": [np.nan],
        "lower.CI": [np.nan],
        "upper.1": [np.nan],
        "lower.1": [np.nan],
        "upper.0": [np.nan],
        "lower.0": [np.nan]
    }
    
    return pd.DataFrame(result)