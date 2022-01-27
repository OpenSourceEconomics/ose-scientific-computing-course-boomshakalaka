import numpy as np
import pandas as pd
import us
import os
import gc

from datetime import timedelta
from numpy import linalg as la
from statsmodels.formula.api import ols
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt

### Downloading poll data process
state_name = [state.name for state in us.states.STATES_CONTIGUOUS]
state_dict = {x.abbr: x.name for x in us.states.STATES_CONTIGUOUS}

lambda_ = 0.75
C_1 = np.ones([51, 51])
a = 1
run_date = pd.to_datetime("2016-11-08")
election_day = pd.to_datetime("2016-11-08")
start_date = pd.to_datetime("2016-03-01")

def get_df(input_file):
    df = pd.read_csv(input_file)
    df.drop(columns=["pollster.url", "source.url", "question.text", "question.iteration", "entry.date.time..et.",
                     "partisan", "affiliation", "Unnamed: 0"], inplace=True)

    ### Cleaning up the data ###
    # filtering data

    df.rename(columns={"number.of.observations": "n"}, inplace=True)
    df.rename(columns={"start.date": "start"}, inplace=True)
    df.rename(columns={"end.date": "end"}, inplace=True)
    df["start"] = pd.to_datetime(df["start"], format="%Y-%m-%d")
    df["end"] = pd.to_datetime(df["end"], format="%Y-%m-%d")
    df["t"] = df["end"] - ((timedelta(days=1) + (df["end"] - df["start"])) / 2).dt.ceil("d")
    df = df[(df["t"] >= start_date) & ((df["population"] == "Likely Voters") | (df["population"] == "Registered Voters")
                                       | (df["population"] == "Adults")) & (df["n"] > 1)]

    # pollster arrangements
    characters = "'!^-%&/()=?_.,<$>£#½§{[]}\}|;`"
    for pollster in df["pollster"].unique():
        ch_index_list = []
        for ch in characters:
            ch_index = [i for i, x in enumerate(pollster) if x == ch]
            if ch_index:
                ch_index_list.append(ch_index[0])
        if not ch_index_list:
            continue
        first_ch = min(ch_index_list)
        new_pollster = pollster.split(pollster[first_ch])[0]
        if new_pollster[-1] == " ":
            new_pollster = new_pollster[:-1]
        df.replace(pollster, new_pollster, inplace=True)

    df.replace(["Fox News", "WashPost", "ABC News", "DHM Research", "Public Opinion Strategies"],
               ["FOX", "Washington Post", "ABC", "DHM", "POS"], inplace=True)

    df["mode"].replace(
        ["Internet", "Live Phone", 'IVR/Online', 'Live Phone/Online', 'Automated Phone', 'IVR/Live Phone', 'Mixed',
         'Mail'],
        ["Online Poll", "Live Phone Component", *["Other"] * 6], inplace=True)

    # dropping NAs
    df["undecided"][df["undecided"].isna()] = 0
    df["other"][df["other"].isna()] = 0
    df["johnson"][df["johnson"].isna()] = 0
    df["mcmullin"][df["mcmullin"].isna()] = 0

    # calculating two party poll shares
    df["twoparty"] = df["clinton"] + df["trump"]
    df["polltype"] = df["population"]
    # calculating Clinton vote shares
    df["n_clinton"] = round(df["n"] * df["clinton"] / 100)
    df["pct_clinton"] = df["clinton"] / df["twoparty"]
    # calculating Trump vote shares
    df["n_trump"] = round(df["n"] * df["trump"] / 100)
    df["pct_trump"] = df["trump"] / df["twoparty"]

    # # importing abbrevetions
    # state_abbr = {x.abbr: x.name for x in us.states.STATES_CONTIGUOUS}
    # state_abbr_items = state_abbr.items()
    # state_abbr_list = list(state_abbr_items)
    # state_abbr =pd.DataFrame(state_abbr_list, columns=["Abbr", "State"])
    #
    # a = pd.read_csv("abbr_list.csv")
    # #combining with df
    # df["Abbr"] = np.where(df["state"] == "--", "General", df["state"].map(state_abbr.set_index("State")["Abbr"]))
    df["poll_day"] = df["t"] - min(df["t"]) + timedelta(days=1)

    # creating indexes
    columns = ["state", "pollster", "polltype", "mode"]
    index_i = ["s", "p", "pop", "m"]
    for i, col in enumerate(columns):
        reindex = False
        for ii, x in enumerate(df[col].sort_values().unique(), start=1):
            if reindex:
                ii -= 1
            if x == "--":
                ii = 51
                reindex = True
            df.loc[df[col] == x, f"index_{index_i[i]}"] = ii
        df[f"index_{index_i[i]}"] = df[f"index_{index_i[i]}"].astype(int)

    df["index_t"] = df["poll_day"].dt.days

    df = df.sort_values(by=["state", "t", "polltype", "twoparty"])
    df.drop_duplicates(['state', 't', 'pollster'], inplace=True)

    return df


# day indices
df = get_df("data/all_polls.csv")
first_day = min(df["start"])
ndays = max(df["t"]) - min(df["t"])

all_polled_states = df["state"].unique()
all_polled_states = np.delete(all_polled_states, 0)

# getting states info from 2012
state2012 = pd.read_csv("data/2012.csv")
state2012["score"] = state2012["obama_count"] / (state2012["obama_count"] + state2012["romney_count"])
state2012["national score"] = sum(state2012["obama_count"]) / sum(state2012["obama_count"] + state2012["romney_count"])
state2012["delta"] = state2012["score"] - state2012["national score"]
state2012["share_national_vote"] = (state2012["total_count"] * (1 + state2012["adult_pop_growth_2011_15"])) \
                                   / sum(state2012["total_count"] * (1 + state2012["adult_pop_growth_2011_15"]))
state2012 = state2012.sort_values("state")

state_abb = state2012["state"]
state_name = state2012["state_name"]

prior_diff_score = pd.DataFrame(state2012["delta"])
prior_diff_score.set_index(state_abb, inplace=True)

state_weights = pd.DataFrame(state2012["share_national_vote"] / sum(state2012["share_national_vote"]))
state_weights.set_index(state_abb.sort_values(), inplace=True)

##creating covariance matrices
# preparing data
state_data = pd.read_csv("data/abbr_list.csv")
state_data = state_data[["year", "state", "dem"]]
state_data = state_data[state_data["year"] == 2016]
state_data.rename(columns={"year": "variable", "dem": "value"}, inplace=True)
state_data = state_data[["state", "variable", "value"]]

census = pd.read_csv("data/acs_2013_variables.csv")
census.dropna(inplace=True)
census.drop(columns=["state_fips", "pop_total", "pop_density"], inplace=True)
census = census.melt(id_vars="state")

state_data = state_data.append(census)

# adding urbanicity
urbanicity = pd.read_csv("data/urbanicity_index.csv")
urbanicity.rename(columns={"average_log_pop_within_5_miles": "pop_density"}, inplace=True)
urbanicity = urbanicity[["state", "pop_density"]]
urbanicity = urbanicity.melt(id_vars="state")

state_data = state_data.append(urbanicity)

# adding white evangelical
white_pct = pd.read_csv("data/white_evangel_pct.csv")
white_pct = white_pct.melt(id_vars="state")

state_data = state_data.append(white_pct)

# adding regions as dummy
regions = pd.read_csv("data/state_region_crosswalk.csv")
regions.rename(columns={"state_abb": "state", "region": "variable"}, inplace=True)
regions["value"] = 1
regions = regions[["state", "variable", "value"]]
regions = regions.pivot_table(index="state", columns="variable", values="value", fill_value=0).reset_index("state")
regions = regions.melt(id_vars="state")

# spread the data
state_data_long = state_data.copy()
state_data_long["value"] = state_data_long.groupby("variable")["value"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()))
state_data_long = state_data_long.pivot_table(index="variable", columns="state", values="value").reset_index("variable")
state_data_long.drop(columns=["variable"], inplace=True)

# creting and computing correlation matrix
# formula : a*(lambda*C + (1-lambda)*C_1)
# where C is corr matrix with min 0
# C_1 is sq matrix with all numbers 1
# lambda = 0 -> 100% corr, lambda = 1 -> our corr matrix
C = state_data_long.corr()

# make the values of C min 0
C = C.clip(lower=0)

tmp_C = C.copy()
np.fill_diagonal(tmp_C.values, np.nan)
(tmp_C.mean()).mean()


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
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


A = (lambda_ * C + (1 - lambda_) * C_1)
new_C = nearestPD(A)

# making positive definite
state_correlation_polling = new_C
state_correlation_polling = nearestPD(state_correlation_polling)


def cov_matrix(n, sigma2, rho):
    m = np.matrix(np.ones(shape=(n, n)) * np.nan)
    m[np.triu_indices(n)] = rho
    m[np.tril_indices(n)] = rho
    np.fill_diagonal(m, 1)
    nn = np.zeros(shape=(n, n))
    np.fill_diagonal(nn, 1)
    return_ = (sigma2 ** .5 * nn) @ m @ (sigma2 ** .5 * nn)
    return return_


# cov matrix for polling error
state_covariance_polling_bias = cov_matrix(51, 0.078 ** 2, 0.9)
state_covariance_polling_bias = state_covariance_polling_bias * state_correlation_polling

np.sqrt(state_weights.T @ state_covariance_polling_bias @ state_weights) / 4

# cov matrix for prior election day prediction
state_covariance_mu_b_T = cov_matrix(51, 0.18 ** 2, 0.9)
state_covariance_mu_b_T = state_covariance_mu_b_T * state_correlation_polling

np.sqrt(state_weights.T @ state_covariance_mu_b_T @ state_weights) / 4

# cov matrix for random walks
state_covariance_mu_b_walk = cov_matrix(51, 0.017 ** 2, 0.9)
# demo corrs to fill gaps in state polls
state_covariance_mu_b_walk = state_covariance_mu_b_walk * state_correlation_polling

(np.sqrt(state_weights.T @ state_covariance_mu_b_walk @ state_weights) / 4) * np.sqrt(300)

# Making default cov matrices:
# initial cov matrix
state_covariance_0 = cov_matrix(51, 0.07 ** 2, 0.9)
state_covariance_0 = state_covariance_0 * state_correlation_polling
# national error
np.sqrt(state_weights.T @ state_covariance_0 @ state_weights) / 4

# initial scaling :
national_cov_matrix_error_sd = np.sqrt(state_weights.T @ state_covariance_0 @ state_weights)  # 0.05


# other scales :
def fit_rmse_day_x(x):
    result = 0.03 + (10 ** (-6.6)) * x ** 2
    return result


diffdays_until_election = (election_day - run_date).days
expected_national_mu_b_T_error = fit_rmse_day_x(diffdays_until_election)  # 0.03

polling_bias_scale = 0.013
mu_b_T_scale = expected_national_mu_b_T_error
random_walk_scale = 0.05 / np.sqrt(300)

cov_poll_bias = state_covariance_0 * ((polling_bias_scale / national_cov_matrix_error_sd * 4) ** 2).values[0][0]
cov_mu_b_T = state_covariance_0 * ((mu_b_T_scale / national_cov_matrix_error_sd * 4) ** 2).values[0][0]
cov_mu_b_walk = state_covariance_0 * ((random_walk_scale / national_cov_matrix_error_sd * 4) ** 2).values[0][0]

# creating priors:
abramowitz = pd.read_csv("data/abramowitz_data.csv")
abramowitz = abramowitz[abramowitz["year"] < 2016]
prior_model = ols("incvote ~ juneapp + q2gdp", data=abramowitz).fit()

# make predictions
new_data = {"juneapp": [4], "q2gdp": [1.1]}
national_mu_prior = prior_model.predict(pd.DataFrame(new_data))
# scaling
national_mu_prior = national_mu_prior / 100


def logging(x):
    result = np.log(x / (1 - x))
    return result


prior_diff_score_values = prior_diff_score.values.reshape(51)
mu_b_prior_result = logging(national_mu_prior.values.reshape(1) + prior_diff_score_values)
mu_b_prior = prior_diff_score.copy()
mu_b_prior["delta"] = mu_b_prior_result

# Pooled voters were different from average voters until September
# creating alpha for inconsistency btw national and state polls
score_among_polled = sum(state2012.drop(7)["obama_count"]) / sum(
    state2012.drop(7)["obama_count"] + state2012.drop(7)["romney_count"])
alpha_prior = np.log(state2012["national score"][1] / score_among_polled)

# pollsters that adjust for party
adjusters = ["ABC", "Washington Post", "Ipsos", "Pew", "YouGov", "NBC"]

# creating variables
N_state_polls = df[df["index_s"] != 51].shape[0]
N_national_polls = df[df["index_s"] == 51].shape[0]
T = (election_day - first_day).days
current_T = max(df["poll_day"])
S = 51  # number of states polled
P = len(df["pollster"].unique())
M = len(df["mode"].unique())
Pop = len(df["polltype"].unique())

state = df[df["index_s"] != 51]["index_s"]
day_national = df[df["index_s"] == 51]["poll_day"]
day_state = df[df["index_s"] != 51]["poll_day"]
poll_national = df[df["index_s"] == 51]["index_p"]
poll_state = df[df["index_s"] != 51]["index_p"]
poll_mode_national = df[df["index_s"] == 51]["index_m"]
poll_mode_state = df[df["index_s"] != 51]["index_m"]
poll_pop_national = df[df["index_s"] == 51]["index_pop"]
poll_pop_state = df[df["index_s"] != 51]["index_pop"]

n_democrat_national = df[df["index_s"] == 51]["n_clinton"]
n_democrat_state = df[df["index_s"] != 51]["n_clinton"]
n_two_share_national = df[df["index_s"] == 51].loc[:, ["n_trump", "n_clinton"]].sum(axis=1)
n_two_share_state = df[df["index_s"] != 51].loc[:, ["n_trump", "n_clinton"]].sum(axis=1)
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


def run_stan_model():
    data = {
        "N_national_polls": N_national_polls,
        "N_state_polls": N_state_polls,
        "T": T,
        "S": S,
        "P": P,
        "M": M,
        "Pop": Pop,
        "state": state,
        "state_weights": state_weights.squeeze(),
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
        "n_democrat_national": n_democrat_national.astype(int),
        "n_democrat_state": n_democrat_state.astype(int),
        "n_two_share_national": n_two_share_national.astype(int),
        "n_two_share_state": n_two_share_state.astype(int),
        "sigma_measure_noise_national": sigma_measure_noise_national,
        "sigma_measure_noise_state": sigma_measure_noise_state,
        "mu_b_prior": mu_b_prior.squeeze(),
        "sigma_c": sigma_c,
        "sigma_m": sigma_m,
        "sigma_pop": sigma_pop,
        "sigma_e_bias": sigma_e_bias,
        "state_covariance_0": state_covariance_0,
        "polling_bias_scale": polling_bias_scale,
        "mu_b_T_scale": mu_b_T_scale,
        "random_walk_scale": random_walk_scale
    }

    n_chains = 6
    n_cores = 6
    n_sampling = 500
    n_warmup = 500
    n_refresh = int(n_sampling * 0.1)

    model = CmdStanModel(stan_file="/home/admin/gözdeproject/poll_model_2020.stan", compile=True)
    fit = model.sample(
        data=data,
        seed=1843,
        parallel_chains=n_cores,
        chains=n_chains,
        iter_warmup=n_warmup,
        iter_sampling=n_sampling,
        refresh=n_refresh
    )
    save_fit = fit.draws_pd().to_csv("data/all_predictions.csv")
    gc.collect()


def logit(p):
    return np.log(p) - np.log(1 - p)


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def mean_low_high(draws, states, id):
    if type(draws) == np.ndarray:
        m = draws.mean(axis=0)
        sd = draws.std(axis=0)
    else:
        m = draws.mean(axis=0).values
        sd = draws.std(axis=0).values
    draws_df = pd.DataFrame(data={"mean": inv_logit(m),
                                  "high": inv_logit(m + 1.96 * sd),
                                  "low": inv_logit(m - 1.96 * sd),
                                  "state": states,
                                  "type": id})

    return draws_df





###########################
#          MU_b
###########################

def get_plots(plot_type):
    out = pd.read_csv("data/all_predictions.csv")
    import matplotlib.pyplot as plt
    if plot_type == "mu_b":
        y = np.random.multivariate_normal(size=1000, mean=mu_b_prior.values.squeeze(), cov=state_covariance_mu_b_T)
        mu_b_T_posterior_draw = out[[x for x in out.columns if "mu_b[" in x and "raw" not in x and "252" in x]]
        mu_b_T_prior_draws = mean_low_high(y, state_name.values, "prior")
        mu_b_T_posterior_draws = mean_low_high(mu_b_T_posterior_draw, state_name.values, "posterior")
        mu_b_T = mu_b_T_prior_draws.append(mu_b_T_posterior_draws)
        mu_b_T = mu_b_T.sort_values(by=["mean", "state"])

        groups = mu_b_T.groupby("type")

        plt.figure(figsize=(8, 12))
        for label, group in groups:
            err = group["high"] - group["low"]
            plt.errorbar(x=group["mean"], y=group["state"], xerr=err, label=label, fmt="o")
        plt.xlabel("Mean")
        plt.axvline(0.5, linestyle="--")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.legend()
        plt.show()
        return

    ###########################
    #          MU_C
    ###########################
    if plot_type == "mu_c":
        mu_c_cols = [x for x in out.columns.values if "mu_c" in x and "raw" not in x]
        mu_c_posterior_draw = out[mu_c_cols].copy()

        pollster_ = df[["pollster", "index_p"]].drop_duplicates().sort_values(by="pollster").values.tolist() * 3000
        pollster = [x[0] for x in pollster_]
        mu_c_posterior_draws = pd.DataFrame({"draws": mu_c_posterior_draw.__array__().reshape(3000 * 162),
                                             "index_p": list(range(1, 163)) * 3000,
                                             "pollster": pollster,
                                             "type": "posterior"})
        pollster_ = df[["pollster", "index_p"]].sort_values(by="pollster").drop_duplicates().values.tolist() * 1000
        pollster = [x[0] for x in pollster_]
        mu_c_prior_draws = pd.DataFrame({"draws": np.random.normal(0, sigma_c, P * 1000),
                                         "index_p": list(range(1, 163)) * 1000,
                                         "pollster": pollster,
                                         "type": "prior"})
        mu_c_draws = mu_c_posterior_draws.append(mu_c_prior_draws)
        mu_c_draws.reset_index(drop=True, inplace=True)

        g = mu_c_draws.groupby(["pollster", "type"])["draws"]
        mu_c_draws = pd.DataFrame({"mean": g.mean(),
                                   "low": g.mean() - 1.96 * g.std(),
                                   "high": g.mean() + 1.96 * g.std()})

        filtered_pollster = df.groupby("pollster").count()["n"]
        filtered_pollster = filtered_pollster[filtered_pollster >= 5].index.tolist()
        mu_c_draws.reset_index(drop=False, inplace=True)
        mu_c_draws_filtered = mu_c_draws.loc[mu_c_draws["pollster"].isin(filtered_pollster)]
        groups = mu_c_draws_filtered.reset_index().sort_values(by=["mean", "pollster"]).groupby("type")

        plt.figure(figsize=(10, 20))
        for label, group in groups:
            # group.sort_values(by="mean", inplace=True)
            err = group["high"] - group["low"]
            plt.errorbar(x=group["mean"], y=group["pollster"], xerr=err, label=label, fmt="o")
        plt.xlabel("Mean")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return
    ###########################
    #          MU_m
    ###########################
    if plot_type == "mu_m":
        mu_m_cols = [x for x in out.columns.values if "mu_m" in x and "raw" not in x]
        mu_m_posterior_draws = out[mu_m_cols].copy()

        method = df[["mode", "index_m"]].sort_values(by="mode").drop_duplicates().values.tolist() * 3000
        method = [x[0] for x in method]
        mu_m_posterior_draws = pd.DataFrame({"draws": mu_m_posterior_draws.__array__().reshape(3000 * 3, ),
                                             "index_m": list(range(1, M + 1)) * mu_m_posterior_draws.shape[0],
                                             "type": "posterior",
                                             "method": method})

        method = df[["mode", "index_m"]].sort_values(by="mode").drop_duplicates().values.tolist() * 1000
        method = [x[0] for x in method]
        mu_m_prior_draws = pd.DataFrame({"draws": np.random.normal(0, sigma_c, M * 1000),
                                         "index_m": list(range(1, M + 1)) * 1000,
                                         "type": "prior",
                                         "method": method})
        mu_m_draws = mu_m_posterior_draws.append(mu_m_prior_draws)

        mu_m_draws.reset_index(drop=True).sort_values(by="index_m", inplace=True)

        g = mu_m_draws.groupby(["method", "type"])["draws"]
        mu_m_draws = pd.DataFrame({"mean": g.mean(),
                                   "low": g.mean() - 1.96 * g.std(),
                                   "high": g.mean() + 1.96 * g.std()})

        groups = mu_m_draws.reset_index(drop=False).sort_values(by=["mean", "method"]).groupby("type")

        plt.figure(figsize=(6, 8))
        for label, group in groups:
            err = group["high"] - group["low"]
            plt.errorbar(x=group["mean"], y=group["method"], xerr=err, label=label, fmt="o")
        plt.xlabel("Mean")
        plt.tight_layout()
        plt.legend()
        plt.show()
        return

    ###########################
    #          MU_pop
    ###########################
    if plot_type == "mu_pop":
        mu_pop_cols = [x for x in out.columns.values if "mu_pop" in x and "raw" not in x]
        mu_pop_posterior_draws = out[mu_pop_cols].copy()

        method = df[["polltype", "index_pop"]].drop_duplicates().values.tolist() * 3000
        method = [x[0] for x in method]
        mu_pop_posterior_draws = pd.DataFrame({"draws": mu_pop_posterior_draws.__array__().reshape(3000 * 3, ),
                                               "index_pop": list(range(1, M + 1)) * mu_pop_posterior_draws.shape[0],
                                               "type": "posterior",
                                               "method": method})

        method = df[["polltype", "index_pop"]].drop_duplicates().values.tolist() * 1000
        method = [x[0] for x in method]
        mu_pop_prior_draws = pd.DataFrame({"draws": np.random.normal(0, sigma_c, M * 1000),
                                           "index_pop": list(range(1, Pop + 1)) * 1000,
                                           "type": "prior",
                                           "method": method})
        mu_pop_draws = mu_pop_posterior_draws.append(mu_pop_prior_draws)

        mu_pop_draws.reset_index(drop=True).sort_values(by="index_pop", inplace=True)

        g = mu_pop_draws.groupby(["method", "type"])["draws"]
        mu_pop_draws = pd.DataFrame({"mean": g.mean(),
                                     "low": g.mean() - 1.96 * g.std(),
                                     "high": g.mean() + 1.96 * g.std()})

        groups = mu_pop_draws.reset_index(drop=False).sort_values(by=["mean", "method"]).groupby("type")

        plt.figure(figsize=(6, 8))
        for label, group in groups:
            err = group["high"] - group["low"]
            plt.errorbar(x=group["mean"], y=group["method"], xerr=err, label=label, fmt="o")
        plt.xlabel("Mean")
        plt.tight_layout()
        plt.legend()
        plt.show()

    ###########################
    #          Polling_Bias
    ###########################
    if plot_type == "polling_bias":
        polling_bias_posterior = out[[x for x in out.columns.values if "polling_bias[" in x and "raw" not in x]]

        polling_bias_posterior_draws = pd.DataFrame({"draws": polling_bias_posterior.__array__().reshape(
            polling_bias_posterior.shape[0] * polling_bias_posterior.shape[1], ),
            "index_s": list(range(1, S + 1)) * polling_bias_posterior.shape[0],
            "type": "posterior",
            "states": state_name.tolist() * polling_bias_posterior.shape[0]})
        y = np.random.multivariate_normal(size=1000, mean=[0] * S, cov=state_covariance_polling_bias)
        polling_bias_prior_draws = pd.DataFrame({"draws": y.reshape(1000 * S),
                                                 "index_s": list(range(1, S + 1)) * 1000,
                                                 "type": "prior",
                                                 "states": state_name.tolist() * 1000})

        polling_bias_draws = polling_bias_posterior_draws.append(polling_bias_prior_draws)
        polling_bias_draws.reset_index(drop=True, inplace=True)

        g = polling_bias_draws.groupby(["states", "type"])["draws"]
        polling_bias_draws = pd.DataFrame({"mean": g.mean(),
                                           "low": g.mean() - 1.96 * g.std(),
                                           "high": g.mean() + 1.96 * g.std()})

        groups = polling_bias_draws.reset_index(drop=False).sort_values(by=["mean", "states"]).groupby("type")
        plt.figure(figsize=(6, 8))
        for label, group in groups:
            err = group["high"] - group["low"]
            plt.errorbar(x=group["mean"], y=group["states"], xerr=err, label=label, fmt="o")
        plt.xlabel("Mean")
        plt.tight_layout()
        plt.legend()
        plt.show()

    ###########################
    #          E_Bias
    ###########################
    if plot_type == "map":
        e_bias_posterior = out[[x for x in out.columns.values if "e_bias[" in x and "raw" not in x]]

        predicted_score = out[[x for x in out.columns.values if "predicted_score[" in x and "raw" not in x]]
        single_draw = predicted_score[[x for x in predicted_score.columns.values if "252" in x]]

        t_list = [(df.start.min().to_pydatetime() + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(1, 253)]
        pct_clinton = pd.DataFrame({"low": predicted_score.quantile(0.025, axis=0).values,
                                    "high": predicted_score.quantile(0.975, axis=0).values,
                                    "mean": predicted_score.mean(axis=0).values,
                                    "prob": predicted_score[predicted_score > 0.5].mean(axis=0).fillna(0).values,
                                    "state": np.sort(state_name.values.tolist() * 252),
                                    "t": t_list * 51})

        nat = predicted_score.to_numpy().reshape(3000, 252, 51)
        nat_ = np.average(nat, axis=2, weights=state_weights.squeeze()).flatten()
        pct_clinton_natl = pd.DataFrame({"natl_vote": nat_,
                                         "t": t_list * 3000,
                                         "draw": np.sort(list(range(1, 3001)) * 252)})

        groups = pct_clinton_natl.groupby("t")
        l = []
        for key, value in groups:
            l.append(round((value["natl_vote"] > 0.5).sum() / 3000, 2))

        pct_clinton_natl = pd.DataFrame({"low": pct_clinton_natl.groupby("t")["natl_vote"].quantile(0.025).values,
                                         "high": pct_clinton_natl.groupby("t")["natl_vote"].quantile(0.975).values,
                                         "mean": pct_clinton_natl.groupby("t")["natl_vote"].mean().values,
                                         "prob": l,
                                         "state": "--",
                                         "t": t_list})

        pct_all = pct_clinton.append(pct_clinton_natl).fillna(0).reset_index(drop=True)

        v1 = pct_all.loc[pct_all["state"] == "--"]
        v1 = v1.set_index(v1["t"])

        v2 = df[["state", "t", "pct_clinton", "mode"]]  # .loc[df["state"] == "--"]
        v2["t"] = v2["t"].dt.strftime("%Y-%m-%d")
        v2 = v2.set_index(v2["t"])
        v2.index = v2.index.strftime("%Y-%m-%d")

        pct_all_plt = pd.concat([v1, v2])
        pct_all_plt = pct_all_plt.fillna(method='ffill').sort_index()
        # pct_all_plt = pct_all_plt.sort_index()
        # pct_all_plt = pct_all_plt.groupby(level=0).sum()

        plt_clinton = pct_all.loc[pct_all["t"] == "2016-11-08"][["state", "prob"]].reset_index(drop=True)

        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        st_list = set([x["NAME"] for x in map.states_info])
        # create the map
        map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
                      projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

        # load the shapefile, use the name 'states'
        map.readshapefile('/home/admin/Downloads/st99_d00', name='states', drawbounds=True)

        # collect the state names from the shapefile attributes so we can
        # look up the shape obect for a state by it's name
        state_names = []
        for shape_dict in map.states_info:
            state_names.append(shape_dict['NAME'])

        ax = plt.gca()  # get current axes instance

        # get Texas and draw the filled polygon
        colors = plt.get_cmap("RdBu")
        cl = []
        pt = []
        a = 0
        prob_colors = (plt_clinton["prob"] * 255).astype(int).values
        for i in plt_clinton["state"]:
            if "--" != i:
                seg = map.states[state_names.index(i)]
                poly = Polygon(seg, facecolor=colors(prob_colors[a]), edgecolor='red')
                ax.add_patch(poly)
                cl.append(colors(a * 5))
                pt.append(poly)

            a += 1
        p = PatchCollection(pt, cmap=colors)
        p.set_array(np.array(cl))
        cb = plt.colorbar(p)
        plt.show()
        return

    ###########################
    #          States
    ###########################
    if plot_type == "states":
        predicted_score = out[[x for x in out.columns.values if "predicted_score[" in x and "raw" not in x]]
        single_draw = predicted_score[[x for x in predicted_score.columns.values if "252" in x]]

        t_list = [(df.start.min().to_pydatetime() + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(1, 253)]
        pct_clinton = pd.DataFrame({"low": predicted_score.quantile(0.025, axis=0).values,
                                    "high": predicted_score.quantile(0.975, axis=0).values,
                                    "mean": predicted_score.mean(axis=0).values,
                                    "prob": predicted_score[predicted_score > 0.5].mean(axis=0).fillna(0).values,
                                    "state": np.sort(state_name.values.tolist() * 252),
                                    "t": t_list * 51})

        nat = predicted_score.to_numpy().reshape(3000, 252, 51)
        nat_ = np.average(nat, axis=2, weights=state_weights.squeeze()).flatten()
        pct_clinton_natl = pd.DataFrame({"natl_vote": nat_,
                                         "t": t_list * 3000,
                                         "draw": np.sort(list(range(1, 3001)) * 252)})

        groups = pct_clinton_natl.groupby("t")
        l = []
        for key, value in groups:
            l.append(round((value["natl_vote"] > 0.5).sum() / 3000, 2))

        pct_clinton_natl = pd.DataFrame({"low": pct_clinton_natl.groupby("t")["natl_vote"].quantile(0.025).values,
                                         "high": pct_clinton_natl.groupby("t")["natl_vote"].quantile(0.975).values,
                                         "mean": pct_clinton_natl.groupby("t")["natl_vote"].mean().values,
                                         "prob": l,
                                         "state": "--",
                                         "t": t_list})

        pct_all = pct_clinton.append(pct_clinton_natl).fillna(0).reset_index(drop=True)

        v1 = pct_all.loc[pct_all["state"] != "--"]
        v1 = v1.set_index(v1["t"])

        groups = v1.groupby("state")
        names = state_name.values.tolist()
        state_all = {x: state_abb.values.tolist()[i] for i, x in enumerate(names)}
        ix = 0
        plt.figure(figsize=(8, 20))
        plt.subplots(2, 5)
        for key, value in groups:
            if "District" in key:
                continue
            if (ix % 10 == 0 and ix != 0) or ix == 49:
                plt.show()
                ix = 0
                plt.figure(figsize=(8, 20))
                plt.subplots(2, 5, sharex=True, sharey=True)

            plt.subplot(2, 5, (ix + 1))
            plt.fill_between(value.index, value["low"], value["high"], alpha=0.3, color="gray")
            plt.plot(value.index, value["mean"])
            plt.axhline(inv_logit(mu_b_prior.values[state_name.tolist().index(key)]), linestyle="--", color="black")
            plt.box(False)
            plt.xticks(list(range(0, 252, 30)), ["M", "A", "M", "J", "J", "A", "S", "O", "N"])
            plt.title(state_all[key])
            plt.tight_layout()

            ix += 1
        return
