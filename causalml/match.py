import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

logger = logging.getLogger("causalml")


def smd(feature, treatment):
    """Calculate the standard mean difference (SMD) of a feature between the
    treatment and control groups.

    The definition is available at
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/#s11title

    Args:
        feature (pandas.Series): a column of a feature to calculate SMD for
        treatment (pandas.Series): a column that indicate whether a row is in
                                   the treatment group or not

    Returns:
        (float): The SMD of the feature
    """
    t = feature[treatment == 1]
    c = feature[treatment == 0]
    return (t.mean() - c.mean()) / np.sqrt(0.5 * (t.var() + c.var()))


def create_table_one(data, treatment_col, features):
    """Report balance in input features between the treatment and control groups.

    References:
        R's tableone at CRAN: https://github.com/kaz-yos/tableone
        Python's tableone at PyPi: https://github.com/tompollard/tableone

    Args:
        data (pandas.DataFrame): total or matched sample data
        treatment_col (str): the column name for the treatment
        features (list of str): the column names of features

    Returns:
        (pandas.DataFrame): A table with the means and standard deviations in
            the treatment and control groups, and the SMD between two groups
            for the features.
    """
    t1 = pd.pivot_table(
        data[features + [treatment_col]],
        columns=treatment_col,
        aggfunc=[lambda x: "{:.2f} ({:.2f})".format(x.mean(), x.std())],
    )
    t1.columns = t1.columns.droplevel(level=0)
    t1["SMD"] = data[features].apply(lambda x: smd(x, data[treatment_col])).round(4)

    n_row = pd.pivot_table(
        data[[features[0], treatment_col]], columns=treatment_col, aggfunc=["count"]
    )
    n_row.columns = n_row.columns.droplevel(level=0)
    n_row["SMD"] = ""
    n_row.index = ["n"]

    t1 = pd.concat([n_row, t1], axis=0)
    t1.columns.name = ""
    t1.columns = ["Control", "Treatment", "SMD"]
    t1.index.name = "Variable"

    return t1


class NearestNeighborMatch(object):
    """
    Propensity score matching based on the nearest neighbor algorithm.

    Attributes:
        caliper (float): threshold to be considered as a match.
        replace (bool): whether to match with replacement or not
        ratio (int): ratio of control / treatment to be matched. used only if
            replace=True.
        shuffle (bool): whether to shuffle the treatment group data before
            matching
        random_state (numpy.random.RandomState or int): RandomState or an int
            seed
        n_jobs (int): The number of parallel jobs to run for neighbors search.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors
    """

    def __init__(
        self,
        caliper=0.2,
        replace=False,
        ratio=1,
        shuffle=True,
        random_state=None,
        n_jobs=-1,
    ):
        """Initialize a propensity score matching model.

        Args:
            caliper (float): threshold to be considered as a match.
            replace (bool): whether to match with replacement or not
            shuffle (bool): whether to shuffle the treatment group data before
                matching or not
            random_state (numpy.random.RandomState or int): RandomState or an
                int seed
            n_jobs (int): The number of parallel jobs to run for neighbors search.
                None means 1 unless in a joblib.parallel_backend context. -1 means using all processors
        """
        self.caliper = caliper
        self.replace = replace
        self.ratio = ratio
        self.shuffle = shuffle
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs

    def match(self, data, treatment_col, score_cols):
        """Find matches from the control group by matching on specified columns
        (propensity preferred).

        Args:
            data (pandas.DataFrame): total input data
            treatment_col (str): the column name for the treatment
            score_cols (list): list of column names for matching (propensity
                column should be included)

        Returns:
            (pandas.DataFrame): The subset of data consisting of matched
                treatment and control group data.
        """
        assert type(score_cols) is list, "score_cols must be a list"
        treatment = data.loc[data[treatment_col] == 1, score_cols]
        control = data.loc[data[treatment_col] == 0, score_cols]

        sdcal = self.caliper * np.std(data[score_cols].values)

        if self.replace:
            scaler = StandardScaler()
            scaler.fit(data[score_cols])
            treatment_scaled = pd.DataFrame(
                scaler.transform(treatment), index=treatment.index
            )
            control_scaled = pd.DataFrame(
                scaler.transform(control), index=control.index
            )

            # SD is the same as caliper because we use a StandardScaler above
            sdcal = self.caliper

            matching_model = NearestNeighbors(
                n_neighbors=self.ratio, n_jobs=self.n_jobs
            )
            matching_model.fit(control_scaled)
            distances, indices = matching_model.kneighbors(treatment_scaled)

            # distances and indices are (n_obs, self.ratio) matrices.
            # To index easily, reshape distances, indices and treatment into
            # the (n_obs * self.ratio, 1) matrices and data frame.
            distances = distances.T.flatten()
            indices = indices.T.flatten()
            treatment_scaled = pd.concat([treatment_scaled] * self.ratio, axis=0)

            cond = (distances / np.sqrt(len(score_cols))) < sdcal
            # Deduplicate the indices of the treatment group
            t_idx_matched = np.unique(treatment_scaled.loc[cond].index)
            # XXX: Should we deduplicate the indices of the control group too?
            c_idx_matched = np.array(control_scaled.iloc[indices[cond]].index)
        else:
            assert len(score_cols) == 1, (
                "Matching on multiple columns is only supported using the "
                "replacement method (if matching on multiple columns, set "
                "replace=True)."
            )
            # unpack score_cols for the single-variable matching case
            score_col = score_cols[0]

            if self.shuffle:
                t_indices = self.random_state.permutation(treatment.index)
            else:
                t_indices = treatment.index

            t_idx_matched = []
            c_idx_matched = []
            control["unmatched"] = True

            for t_idx in t_indices:
                dist = np.abs(
                    control.loc[control.unmatched, score_col]
                    - treatment.loc[t_idx, score_col]
                )
                c_idx_min = dist.idxmin()
                if dist[c_idx_min] <= sdcal:
                    t_idx_matched.append(t_idx)
                    c_idx_matched.append(c_idx_min)
                    control.loc[c_idx_min, "unmatched"] = False

        return data.loc[
            np.concatenate([np.array(t_idx_matched), np.array(c_idx_matched)])
        ]

    def match_by_group(self, data, treatment_col, score_cols, groupby_col):
        """Find matches from the control group stratified by groupby_col, by
        matching on specified columns (propensity preferred).

        Args:
            data (pandas.DataFrame): total sample data
            treatment_col (str): the column name for the treatment
            score_cols (list): list of column names for matching (propensity
                column should be included)
            groupby_col (str): the column name to be used for stratification

        Returns:
            (pandas.DataFrame): The subset of data consisting of matched
                treatment and control group data.
        """
        matched = data.groupby(groupby_col).apply(
            lambda x: self.match(
                data=x, treatment_col=treatment_col, score_cols=score_cols
            )
        )
        return matched.reset_index(level=0, drop=True)


class MatchOptimizer(object):
    def __init__(
        self,
        treatment_col="is_treatment",
        ps_col="pihat",
        user_col=None,
        matching_covariates=["pihat"],
        max_smd=0.1,
        max_deviation=0.1,
        caliper_range=(0.01, 0.5),
        max_pihat_range=(0.95, 0.999),
        max_iter_per_param=5,
        min_users_per_group=1000,
        smd_cols=["pihat"],
        dev_cols_transformations={"pihat": np.mean},
        dev_factor=1.0,
        verbose=True,
    ):
        """Finds the set of parameters that gives the best matching result.

        Score = (number of features with SMD > max_smd)
                + (sum of deviations for important variables
                   * deviation factor)

        The logic behind the scoring is that we are most concerned with
        minimizing the number of features where SMD is lower than a certain
        threshold (max_smd). However, we would also like the matched dataset
        not deviate too much from the original dataset, in terms of key
        variable(s), so that we still retain a similar userbase.

        Args:
            - treatment_col (str): name of the treatment column
            - ps_col (str): name of the propensity score column
            - max_smd (float): maximum acceptable SMD
            - max_deviation (float): maximum acceptable deviation for
                important variables
            - caliper_range (tuple): low and high bounds for caliper search
                range
            - max_pihat_range (tuple): low and high bounds for max pihat
                search range
            - max_iter_per_param (int): maximum number of search values per
                parameters
            - min_users_per_group (int): minimum number of users per group in
                matched set
            - smd_cols (list): score is more sensitive to these features
                exceeding max_smd
            - dev_factor (float): importance weight factor for dev_cols
                (e.g. dev_factor=1 means a 10% deviation leads to penalty of 1
                in score)
            - dev_cols_transformations (dict): dict of transformations to be
                made on dev_cols
            - verbose (bool): boolean flag for printing statements

        Returns:
            The best matched dataset (pd.DataFrame)
        """
        self.treatment_col = treatment_col
        self.ps_col = ps_col
        self.user_col = user_col
        self.matching_covariates = matching_covariates
        self.max_smd = max_smd
        self.max_deviation = max_deviation
        self.caliper_range = np.linspace(*caliper_range, num=max_iter_per_param)
        self.max_pihat_range = np.linspace(*max_pihat_range, num=max_iter_per_param)
        self.max_iter_per_param = max_iter_per_param
        self.min_users_per_group = min_users_per_group
        self.smd_cols = smd_cols
        self.dev_factor = dev_factor
        self.dev_cols_transformations = dev_cols_transformations
        self.best_params = {}
        self.best_score = 1e7  # ideal score is 0
        self.verbose = verbose
        self.pass_all = False

    def single_match(self, score_cols, pihat_threshold, caliper):
        matcher = NearestNeighborMatch(caliper=caliper, replace=True)
        df_matched = matcher.match(
            data=self.df[self.df[self.ps_col] < pihat_threshold],
            treatment_col=self.treatment_col,
            score_cols=score_cols,
        )
        return df_matched

    def check_table_one(self, tableone, matched, score_cols, pihat_threshold, caliper):
        # check if better than past runs
        smd_values = np.abs(tableone[tableone.index != "n"]["SMD"].astype(float))
        num_cols_over_smd = (smd_values >= self.max_smd).sum()
        self.cols_to_fix = (
            smd_values[smd_values >= self.max_smd]
            .sort_values(ascending=False)
            .index.values
        )
        if self.user_col is None:
            num_users_per_group = (
                matched.reset_index().groupby(self.treatment_col)["index"].count().min()
            )
        else:
            num_users_per_group = (
                matched.groupby(self.treatment_col)[self.user_col].count().min()
            )
        deviations = [
            np.abs(
                self.original_stats[col]
                / matched[matched[self.treatment_col] == 1][col].mean()
                - 1
            )
            for col in self.dev_cols_transformations.keys()
        ]

        score = num_cols_over_smd
        score += len(
            [col for col in self.smd_cols if smd_values.loc[col] >= self.max_smd]
        )
        score += np.sum([dev * 10 * self.dev_factor for dev in deviations])

        # check if can be considered as best score
        if score < self.best_score and num_users_per_group > self.min_users_per_group:
            self.best_score = score
            self.best_params = {
                "score_cols": score_cols.copy(),
                "pihat": pihat_threshold,
                "caliper": caliper,
            }
            self.best_matched = matched.copy()
        if self.verbose:
            logger.info(
                "\tScore: {:.03f} (Best Score: {:.03f})\n".format(
                    score, self.best_score
                )
            )

        # check if passes all criteria
        self.pass_all = (
            (num_users_per_group > self.min_users_per_group)
            and (num_cols_over_smd == 0)
            and all(dev < self.max_deviation for dev in deviations)
        )

    def match_and_check(self, score_cols, pihat_threshold, caliper):
        if self.verbose:
            logger.info(
                "Preparing match for: caliper={:.03f}, "
                "pihat_threshold={:.03f}, "
                "score_cols={}".format(caliper, pihat_threshold, score_cols)
            )
        df_matched = self.single_match(
            score_cols=score_cols, pihat_threshold=pihat_threshold, caliper=caliper
        )
        tableone = create_table_one(
            df_matched, self.treatment_col, self.matching_covariates
        )
        self.check_table_one(tableone, df_matched, score_cols, pihat_threshold, caliper)

    def search_best_match(self, df):
        self.df = df

        self.original_stats = {}
        for col, trans in self.dev_cols_transformations.items():
            self.original_stats[col] = trans(
                self.df[self.df[self.treatment_col] == 1][col]
            )

        # search best max pihat
        if self.verbose:
            logger.info("SEARCHING FOR BEST PIHAT")
        score_cols = [self.ps_col]
        caliper = self.caliper_range[-1]
        for pihat_threshold in self.max_pihat_range:
            self.match_and_check(score_cols, pihat_threshold, caliper)

        # search best score_cols
        if self.verbose:
            logger.info("SEARCHING FOR BEST SCORE_COLS")
        pihat_threshold = self.best_params["pihat"]
        caliper = self.caliper_range[int(self.caliper_range.shape[0] / 2)]
        score_cols = [self.ps_col]
        while not self.pass_all:
            if len(self.cols_to_fix) == 0:
                break
            elif np.intersect1d(self.cols_to_fix, score_cols).shape[0] > 0:
                break
            else:
                score_cols.append(self.cols_to_fix[0])
                self.match_and_check(score_cols, pihat_threshold, caliper)

        # search best caliper
        if self.verbose:
            logger.info("SEARCHING FOR BEST CALIPER")
        score_cols = self.best_params["score_cols"]
        pihat_threshold = self.best_params["pihat"]
        for caliper in self.caliper_range:
            self.match_and_check(score_cols, pihat_threshold, caliper)

        # summarize
        if self.verbose:
            logger.info("\n-----\nBest params are:\n{}".format(self.best_params))

        return self.best_matched


if __name__ == "__main__":

    from .features import TREATMENT_COL, SCORE_COL, GROUPBY_COL, PROPENSITY_FEATURES
    from .features import PROPENSITY_FEATURE_TRANSFORMATIONS, MATCHING_COVARIATES
    from .features import load_data
    from .propensity import ElasticNetPropensityModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, dest="input_file")
    parser.add_argument("--output-file", required=True, dest="output_file")
    parser.add_argument("--treatment-col", default=TREATMENT_COL, dest="treatment_col")
    parser.add_argument("--groupby-col", default=GROUPBY_COL, dest="groupby_col")
    parser.add_argument(
        "--feature-cols", nargs="+", default=PROPENSITY_FEATURES, dest="feature_cols"
    )
    parser.add_argument("--caliper", type=float, default=0.2)
    parser.add_argument("--replace", default=False, action="store_true")
    parser.add_argument("--ratio", type=int, default=1)

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger.info("Loading data from {}".format(args.input_file))
    df = pd.read_csv(args.input_file)
    df[args.treatment_col] = df[args.treatment_col].astype(int)
    logger.info("shape: {}\n{}".format(df.shape, df.head()))

    pm = ElasticNetPropensityModel(random_state=42)
    w = df[args.treatment_col].values
    X = load_data(
        data=df,
        features=args.feature_cols,
        transformations=PROPENSITY_FEATURE_TRANSFORMATIONS,
    )

    logger.info("Scoring with a propensity model: {}".format(pm))
    df[SCORE_COL] = pm.fit_predict(X, w)

    logger.info(
        "Balance before matching:\n{}".format(
            create_table_one(
                data=df, treatment_col=args.treatment_col, features=MATCHING_COVARIATES
            )
        )
    )
    logger.info(
        "Matching based on the propensity score with the nearest neighbor model"
    )
    psm = NearestNeighborMatch(replace=args.replace, ratio=args.ratio, random_state=42)
    matched = psm.match_by_group(
        data=df,
        treatment_col=args.treatment_col,
        score_cols=[SCORE_COL],
        groupby_col=args.groupby_col,
    )
    logger.info("shape: {}\n{}".format(matched.shape, matched.head()))

    logger.info(
        "Balance after matching:\n{}".format(
            create_table_one(
                data=matched,
                treatment_col=args.treatment_col,
                features=MATCHING_COVARIATES,
            )
        )
    )
    matched.to_csv(args.output_file, index=False)
    logger.info("Matched data saved as {}".format(args.output_file))
