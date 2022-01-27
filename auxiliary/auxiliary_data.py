import numpy as np
import pandas as pd
import us
import requests
from datetime import timedelta


def data_preparation(df):
    df.drop(columns=["pollster.url", "source.url", "question.text", "question.iteration", "entry.date.time..et.",
                     "partisan", "affiliation", "Unnamed: 0"], inplace=True)

    # #adding states to the df
    # for state in state_name:
    #     state2 = state.replace(" ", "-").lower()
    #     df_state = pd.read_csv(f"temporary_{state2}.csv")
    #     df_state["State"] = state
    #     df_state.drop(columns=["Pollster URL", "Source URL", "Question Text", "Entry Date/Time (ET)",
    #                      "Partisan", "Affiliation", "Question Iteration"], inplace=True)
    #     df = df.append(df_state)


    ### Cleaning up the data ###
    #filtering data
    run_date = pd.to_datetime("2016-11-08")
    election_day = pd.to_datetime("2016-11-08")
    start_date = pd.to_datetime("2016-03-01")
    df.rename(columns={"number.of.observations": "n"}, inplace=True)
    df.rename(columns={"start.date": "start"}, inplace=True)
    df.rename(columns={"end.date": "end"}, inplace=True)
    df["start"] = pd.to_datetime(df["start"], format="%Y-%m-%d")
    df["end"] = pd.to_datetime(df["end"], format="%Y-%m-%d")
    df["t"] = df["end"] - ((timedelta(days=1) + (df["end"] - df["start"])) / 2).dt.ceil("d")
    df = df[(df["t"] >= start_date) & ((df["population"] == "Likely Voters") | (df["population"] == "Registered Voters")
            | (df["population"] == "Adults")) & (df["n"] > 1)]

    #pollster arrangements
    characters= "'!^-%&/()=?_.,<$>£#½§{[]}\}|;`"
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

    df["mode"].replace(["Internet", "Live Phone", 'IVR/Online', 'Live Phone/Online', 'Automated Phone', 'IVR/Live Phone', 'Mixed', 'Mail'],
                       ["Online Poll", "Live Phone Component", *["Other"]*6], inplace=True)

    #dropping NAs
    df["undecided"][df["undecided"].isna()] = 0
    df["other"][df["other"].isna()] = 0
    df["johnson"][df["johnson"].isna()] = 0
    df["mcmullin"][df["mcmullin"].isna()] = 0


    #calculating two party poll shares
    df["twoparty"] = df["clinton"] + df["trump"]
    df["polltype"] = df["population"]
    #calculating Clinton vote shares
    df["n_clinton"] = round(df["n"] * df["clinton"] / 100)
    df["pct_clinton"] = df["clinton"] / df["twoparty"]
    #calculating Trump vote shares
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

    #creating indexes
    columns = ["state", "pollster", "polltype", "mode"]
    index_i = ["s", "p", "pop", "m"]
    for i, col in enumerate(columns):
        reindex = False
        for ii, x in enumerate(df[col].sort_values().unique(), start=1):
            if reindex:
                ii-=1
            if x=="--":
                ii=51
                reindex=True
            df.loc[df[col] == x, f"index_{index_i[i]}"] = ii
        df[f"index_{index_i[i]}"] = df[f"index_{index_i[i]}"].astype(int)

    df["index_t"] = df["poll_day"].dt.days

    #sorting and dropping duplicates
    df = df.sort_values(by=["state", "t", "polltype", "twoparty" ])
    df.drop_duplicates(['state','t','pollster'], inplace=True)
    return df

