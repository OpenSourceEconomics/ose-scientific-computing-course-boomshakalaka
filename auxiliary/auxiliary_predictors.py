import numpy as np
import pandas as pd

def correlators(x):

    #creting and computing correlation matrix
    #formula : a*(lambda*C + (1-lambda)*C_1)
    #where C is corr matrix with min 0
    #C_1 is sq matrix with all numbers 1
    #lambda = 0 -> 100% corr, lambda = 1 -> our corr matrix
    C = state_data_long.corr()

    #make the values of C min 0
    C = C.clip(lower=0)

    tmp_C = C.copy()
    np.fill_diagonal(tmp_C.values, np.nan)
    (tmp_C.mean()).mean()

    lambda_ = 0.75
    C_1 = np.ones([51,51])
    a = 1

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

A = (lambda_*C + (1-lambda_)*C_1)
new_C = nearestPD(A)

#making positive definite
state_correlation_polling = new_C
state_correlation_polling = nearestPD(state_correlation_polling)

def cov_matrix(n, sigma2, rho):
    m = np.matrix(np.ones(shape=(n,n)) * np.nan)
    m[np.triu_indices(n)] = rho
    m[np.tril_indices(n)] = rho
    np.fill_diagonal(m, 1)
    nn = np.zeros(shape=(n, n))
    np.fill_diagonal(nn, 1)
    return_ = (sigma2**.5 * nn) @ m @ (sigma2**.5 * nn)
    return return_

def state_covariances(c):
    #cov matrix for polling error
    state_covariance_polling_bias = cov_matrix(51, 0.078**2, 0.9)
    state_covariance_polling_bias = state_covariance_polling_bias * state_correlation_polling

    np.sqrt(state_weights.T @ state_covariance_polling_bias @ state_weights) / 4

    #cov matrix for prior election day prediction
    state_covariance_mu_b_T = cov_matrix(51, 0.18**2, 0.9)
    state_covariance_mu_b_T = state_covariance_mu_b_T * state_correlation_polling

    np.sqrt(state_weights.T @ state_covariance_mu_b_T @ state_weights) / 4

    #cov matrix for random walks
    state_covariance_mu_b_walk = cov_matrix(51, 0.017**2, 0.9)
    #demo corrs to fill gaps in state polls
    state_covariance_mu_b_walk = state_covariance_mu_b_walk * state_correlation_polling

    (np.sqrt(state_weights.T @ state_covariance_mu_b_walk @ state_weights) / 4) * np.sqrt(300)

    #Making default cov matrices:
    #initial cov matrix
    state_covariance_0 = cov_matrix(51, 0.07**2, 0.9)
    state_covariance_0 = state_covariance_0 * state_correlation_polling
    #national error
    np.sqrt(state_weights.T @ state_covariance_0 @ state_weights) / 4

    #initial scaling :
    national_cov_matrix_error_sd = np.sqrt(state_weights.T @ state_covariance_0 @ state_weights) #0.05

#other scales :
def fit_rmse_day_x(x):
    result = 0.03 + (10**(-6.6)) * x**2
    return result

def days(x,y):
    diffdays_until_election = (election_day-run_date).days
    expected_national_mu_b_T_error = fit_rmse_day_x(diffdays_until_election) #0.03

    polling_bias_scale = 0.013
    mu_b_T_scale = expected_national_mu_b_T_error
    random_walk_scale = 0.05/np.sqrt(300)

    cov_poll_bias = state_covariance_0 * ((polling_bias_scale/national_cov_matrix_error_sd*4)**2).values[0][0]
    cov_mu_b_T = state_covariance_0 * ((mu_b_T_scale/national_cov_matrix_error_sd*4)**2).values[0][0]
    cov_mu_b_walk = state_covariance_0 * ((random_walk_scale/national_cov_matrix_error_sd*4)**2).values[0][0]

def priors(df):
    #creating priors:
    abramowitz = pd.read_csv("abramowitz_data.csv")
    abramowitz =abramowitz[abramowitz["year"] < 2016]
    prior_model = ols("incvote ~ juneapp + q2gdp", data = abramowitz).fit()

    #make predictions
    new_data= {"juneapp":[4], "q2gdp":[1.1]}
    national_mu_prior = prior_model.predict(pd.DataFrame(new_data))
    #scaling
    national_mu_prior = national_mu_prior / 100

def logging(x):
    result = np.log(x/(1-x))
    return result


def score_values(x):
    prior_diff_score_values = prior_diff_score.values.reshape(51)
    mu_b_prior_result = logging(national_mu_prior.values.reshape(1) + prior_diff_score_values )
    mu_b_prior = prior_diff_score.copy()
    mu_b_prior["delta"] = mu_b_prior_result

    #Pooled voters were different from average voters until September
    #creating alpha for inconsistency btw national and state polls
    score_among_polled = sum(state2012.drop(7)["obama_count"])/sum(state2012.drop(7)["obama_count"] + state2012.drop(7)["romney_count"])
    alpha_prior = np.log(state2012["national score"][1]/score_among_polled)

    # pollsters that adjust for party
    adjusters = ["ABC", "Washington Post", "Ipsos", "Pew", "YouGov", "NBC"]

    #creating variables
    N_state_polls = df[df["index_s"] != 51].shape[0]
    N_national_polls = df[df["index_s"] == 51].shape[0]
    T = (election_day - first_day).days
    current_T = max(df["poll_day"])
    S = 51 #number of states polled
    P = len(df["pollster"].unique())
    M = len(df["mode"].unique())
    Pop = len(df["polltype"].unique())

    state =  df[df["index_s"] != 51]["index_s"]
    day_national = df[df["index_s"] == 51]["poll_day"]
    day_state =  df[df["index_s"] != 51]["poll_day"]
    poll_national = df[df["index_s"] == 51]["index_p"]
    poll_state =  df[df["index_s"] != 51]["index_p"]
    poll_mode_national = df[df["index_s"] == 51]["index_m"]
    poll_mode_state =  df[df["index_s"] != 51]["index_m"]
    poll_pop_national = df[df["index_s"] == 51]["index_pop"]
    poll_pop_state =  df[df["index_s"] != 51]["index_pop"]

    n_democrat_national = df[df["index_s"] == 51]["n_clinton"]
    n_democrat_state = df[df["index_s"] != 51]["n_clinton"]
    n_two_share_national = df[df["index_s"] == 51].loc[:,["n_trump","n_clinton"]].sum(axis=1)
    n_two_share_state = df[df["index_s"] != 51].loc[:,["n_trump","n_clinton"]].sum(axis=1)
    df["unadjusted"] = np.where(df["pollster"].isin(adjusters), 0, 1)
    unadjusted_national = df[df["index_s"] == 51]["unadjusted"]
    unadjusted_state = df[df["index_s"] != 51]["unadjusted"]

    sigma_measure_noise_national = 0.04
    sigma_measure_noise_state = 0.04
    sigma_c = 0.06
    sigma_m = 0.04
    sigma_pop = 0.04
    sigma_e_bias = 0.02

    polling_bias_scale = float(polling_bias_scale) * 4
    mu_b_T_scale = float(mu_b_T_scale) * 4
    random_walk_scale = float(random_walk_scale) * 4

data = {
    "N_national_polls": N_national_polls,
    "N_state_polls": N_state_polls,
    "T": T,
    "S": S,
    "P": P,
    "M": M,
    "Pop": Pop,
    "state": state,
    "state_weights": state_weights,
    "day_state": day_state.dt.days,
    "day_national": day_national.dt.days,
    "poll_state": poll_state,
    "poll_national": poll_national,
    "poll_mode_national": poll_mode_national,
    "poll_mode_state": poll_mode_state,
    "poll_pop_national": poll_pop_national,
    "poll_pop_state": poll_pop_state,
    "unadjusted_national": unadjusted_national,
    "unadjusted_state": unadjusted_state,
    "n_democrat_national": n_democrat_national,
    "n_democrat_state": n_democrat_state,
    "n_two_share_national": n_two_share_national,
    "n_two_share_state": n_two_share_state,
    "sigma_measure_noise_national": sigma_measure_noise_national,
    "sigma_measure_noise_state": sigma_measure_noise_state,
    "mu_b_prior": mu_b_prior,
    "sigma_c": sigma_c,
    "sigma_m": sigma_m,
    "sigma_pop": sigma_pop,
    "sigma_e_bias": sigma_e_bias,
    "state_covariance_0": state_covariance_0,
    "polling_bias_scale": polling_bias_scale,
    "mu_b_T_scale": mu_b_T_scale,
    "random_walk_scale": random_walk_scale
}
