from typing import List, Union
from copy import deepcopy
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats

from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from causalml.match import create_table_one
# from causalml.match import NearestNeighborMatch
# from causalml.propensity import ElasticNetPropensityModel
# from pymatch.Matcher import Matcher

from pulsar.discovery.distributions import EmpiricalDistribution
from pulsar.visualization import plot_metric_distribution_by_classes
from pulsar.visualization.constants import TREATMENT_COLOR, CONTROL_COLOR


def ks_2sample_bootstrap(x, y, n_iterations: int = 1000, seed: int = None):
    nx = len(x)
    w = np.concatenate((x, y))
    n_total = len(w)
    cutp = nx
    counter = 0
    fs_ks, _ = scipy.stats.ks_2samp(x, y)  # KS-stat, pvalue
    if seed is not None:
        np.random.seed(seed)
    for _ in range(n_iterations):
        sw = np.random.choice(w, n_total, replace=True)
        x1 = sw[:cutp]
        x2 = sw[cutp:]
        s_ks, _ = scipy.stats.ks_2samp(x1, x2)  # KS-stat, pvalue
        if s_ks >= fs_ks:
            counter += 1
    ks_boot_pval = counter * 1.0 / n_iterations
    return ks_boot_pval


class PropensityScoreMatcher:
    def __init__(self):
        self.treatment_color = TREATMENT_COLOR
        self.control_color = CONTROL_COLOR

    def explore_tolerance_threshold(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        score_name: Union[str, List[str]],
        grid: np.array = None,
        x_log: bool = True,
        **kwargs
    ):

        if grid is None:
            # grid = np.linspace(0., 2., 11)[1:]
            grid = np.logspace(-5, 1, 7)
            plt_grid = np.log10(grid)
            print(f'using default grid: {grid}')
        else:
            plt_grid = np.log10(grid) if x_log else grid

        treatment_color = self.treatment_color
        control_color = self.control_color

        treatment, control = deepcopy(data.query(f'{treatment_name} == 1')), deepcopy(data.query(f'{treatment_name} == 0'))
        nt, nc = len(treatment), len(control)

        t_retained, c_retained = [], []
        t_retained_unique, c_retained_unique = [], []

        for tolerance in tqdm(grid):
            kwargs['tolerance'] = tolerance
            kwargs['match_id_name'] = '<match_id>'
            matching_results = self.match(data, treatment_name, score_name, **kwargs)
            matched_data = matching_results['matched_data']
            m_treatment, m_control = deepcopy(matched_data.query(f'{treatment_name} == 1')), deepcopy(matched_data.query(f'{treatment_name} == 0'))
            mt, mc = len(m_treatment), len(m_control)
            mtu, mcu = len(set(m_treatment.index)), len(set(m_control.index))
            t_retained.append(mt / nt)
            t_retained_unique.append(mtu / nt)
            c_retained.append(mc / nc)
            c_retained_unique.append(mcu / nc)

        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(1, 1, figsize = (10, 6))
        ax.plot(plt_grid, t_retained_unique, label = 'treatment unique', color = treatment_color, linestyle = '-', linewidth = 2)
        ax.plot(plt_grid, t_retained, label = 'treatment total', color = treatment_color, linestyle = '-.', linewidth = 2)
        ax.plot(plt_grid, c_retained_unique, label = 'control unique', color = control_color, linestyle = '-', linewidth = 2)
        ax.plot(plt_grid, c_retained, label = 'control total', color = control_color, linestyle = '-.', linewidth = 2)
        ax.set_title("Matched / Initial")
        ax.set_ylabel("Retained ratio")
        ax.set_xlabel("Tolerance threshold" if not x_log else "ln(threshold)")
        ax.legend(loc = 'best')
        plt.close(fig)

        return fig

    @staticmethod
    def match(
        data: pd.DataFrame,
        treatment_name: str,
        score_name: Union[str, List[str]],
        matching_base: str = 'minor',
        method: str = 'min',
        tolerance: float = 0.1,
        force_sd_tolerance: bool = True,
        max_n_matches: int = 1,
        n_neighbors: int = 1,
        scale_before_matching: bool = False,
        p_distance_minkowski: int = 1,
        shuffle: bool = False,
        seed: int = None,
        verbose: bool = True,
        match_id_name: str = 'match_id',
    ):

        """
        one part of `data` based on `treatment_name` becomes 'base' sample and the other one becomes 'other' sample

        matching is based on values in `score_name` which is either a single column name, or a list of several numeric (!) columns in data

        `matching_base` ('minor', 'major', 'treatment', 'control') - strategy determining from which sample to take observations and try to get a pair for it from other sample
        `method` ('min', 'random', 'neighbors') - how to match samples:
        - 'min' goes 1 by 1 in base sample and matches the closest `n_neighbors` from other sample which are still available (`max_n_matches`) and within L1-distance  `tolerance`
        - 'random' goes 1 by 1 in base sample and matches the random `n_neighbors` from other sample which are still available (`max_n_matches`) and within L1-distance `tolerance`
        - 'neighbors' for every observation in base sample tries to simultaneously find `n_neighbors` from other sample (possibly creating many-to-many relations)
        `tolerance` - same as caliper, threshold, bandwidth
         note (!) that it should be stated relative to overall (T+C) scores' standard deviation units, if `force_sd_tolerance` is True,
         unless multicolumn `score_name` provided - then using as is anyway
        `force_sd_tolerance` - determines if tolerance is relational to standard deviation of score values or absolute
        `max_n_matches` - determines how often at most an observation from other sample can be matched to different examples from base sample (not used in 'neighbors' `method`)
        `n_neighbors` - used in all methods, determines how much observations from other sample (counterfactuals) should try to match to observation from base sample
        note that when `n_neighbors` > 1 there still will be different `match_id` for different observations from other sample even when base observation is the same

        `scale_before_matching` - should scores be scaled before distance calculation (could make sense for multiple `score_name` columns)
        `p_distance_minkowski` - kind of distance to use with 'neighbors' `method`

        `shuffle` - determines if order of observations in base sample should be randomly changed before matching
        `seed` - random seed, affects procedure when `shuffle` is True or when using 'random' `method`

        """

        assert len(data) == len(set(data.index)), "for observation's unique identification we're using index from initial data, but they are not unique"
        assert sorted(np.unique(data[treatment_name])) == [0, 1], "treatment flag should be binary integer - 0 or 1"
        assert matching_base in ('minor', 'major', 'treatment', 'control')
        assert method in ('min', 'random', 'neighbors')

        treatment, control = deepcopy(data.query(f'{treatment_name} == 1')), deepcopy(data.query(f'{treatment_name} == 0'))
        nt, nc = len(treatment), len(control)

        if nt <= nc:
            minor, major = treatment, control
            minor_name, major_name = 'treatment', 'control'
        else:
            minor, major = control, treatment
            minor_name, major_name = 'control', 'treatment'

        if matching_base == 'minor':
            base, other = minor, major
            base_name, other_name = minor_name, major_name
        elif matching_base == 'major':
            base, other = major, minor
            base_name, other_name = major_name, minor_name
        elif matching_base == 'treatment':
            base, other = treatment, control
            base_name, other_name = 'treatment', 'control'
        else:
            base, other = control, treatment
            base_name, other_name = 'control', 'treatment'

        if isinstance(score_name, str):
            ### matching will be made on a single column
            base_scaled = deepcopy(base[[score_name]])
            other_scaled = deepcopy(other[[score_name]])
            if force_sd_tolerance:
                sd = np.std(data[score_name], ddof = 1)  ### note that std is calculated on full sample
                sd_tolerance = tolerance * sd
            else:
                sd_tolerance = tolerance
            l = 1
        else:
            ### matching will be made on a multiple columns
            if scale_before_matching:
                scaler = StandardScaler()
                scaler.fit(data[score_name])  ### note that scaling is on full sample
                base_scaled = pd.DataFrame(scaler.transform(base[score_name]), index = base.index)
                other_scaled = pd.DataFrame(scaler.transform(other[score_name]), index = other.index)
                # warnings.warn(f"could only estimate standard deviation for tolerance in matching when score is 1-dimensional, got '{score_name}' as score_name")
            else:
                base_scaled = deepcopy(base[score_name])
                other_scaled = deepcopy(other[score_name])

            sd_tolerance = tolerance
            l = len(score_name)

        if seed is not None:
            np.random.seed(seed)

        if shuffle:
            base_indices = np.random.permutation(base_scaled.index)
        else:
            base_indices = base_scaled.index

        if method == 'neighbors':
            ### nearest neighbors matching with replacement - this part is adapted from causalml.match.NearestNeighborMatch.match
            matching_model = NearestNeighbors(n_neighbors = n_neighbors, p = p_distance_minkowski)
            matching_model.fit(other_scaled) ### fitting on other and remembering coords - from
            distances, indices = matching_model.kneighbors(base_scaled) ### indices are for observations from train sample, both shapes are (n_obs, n_neighbors)
            # to index easily, reshape distances, indices and treatment into
            # the (n_obs * n_neighbors, 1) matrices and dataframe
            # flattening happens by rows first - so the 2 neighbors for single observation from base sample will be separated by n_obs other after flattening
            distances = distances.T.flatten()
            indices = indices.T.flatten()
            condition = (distances / np.sqrt(l)) <= sd_tolerance #### ??? where this sqrt correction comes from ???
            base_scaled = pd.concat([base_scaled] * n_neighbors, axis = 0) ### duplicating base scores in the same way

            # deduplicate_base = True
            # deduplicate_other = False
            # if deduplicate_base:
            #     # deduplicate the indices of the base group
            #     base_idx_matched = np.unique(base_scaled.loc[condition].index)
            # else:
            #     base_idx_matched = base_scaled.loc[condition].index

            # ### ??? should we deduplicate the indices of the control group too ???
            # ### so far it hasnt been done - same observation from other group could be nn for different observations from base group
            # if deduplicate_other:
            #     other_idx_matched = np.unique(other_scaled.iloc[indices[condition]].index)
            # else:
            #     other_idx_matched = np.array(other_scaled.iloc[indices[condition]].index)
            # matched_data = initial_data.loc[np.concatenate([np.array(base_idx_matched), np.array(other_idx_matched)])]

            match_id = 0
            # matches = defaultdict(lambda: defaultdict(int))
            matches = defaultdict(lambda: dict)
            treatment_matches = defaultdict(list)
            control_matches = defaultdict(list)
            for c, b_id, o_id in zip(condition, base_scaled.index, other_scaled.iloc[indices].index):
                if c:
                    match_id += 1
                    matches[match_id] = {base_name: b_id, other_name: o_id}
                    # matches[match_id][base_name] = b_id
                    # matches[match_id][other_name] = o_id
                    if base_name == 'treatment':
                        treatment_matches[b_id].append(match_id)
                        control_matches[o_id].append(match_id)
                    else:
                        treatment_matches[o_id].append(match_id)
                        control_matches[b_id].append(match_id)
        else:
            ### sequential 1 by 1 using min or random sampling from within the tolerance threshold
            assert l == 1, f'method "{method}" allowed only for single score column, got {score_name}'
            other_scaled['matches_available'] = max_n_matches
            match_id = 0
            # matches = defaultdict(lambda: defaultdict(int))
            matches = defaultdict(lambda: dict)
            treatment_matches = defaultdict(list)
            control_matches = defaultdict(list)

            if verbose:
                base_iterator = tqdm(base_indices)
            else:
                base_iterator = base_indices

            for b_id in base_iterator:
                if other_scaled['matches_available'].sum() == 0:
                    break
                # else:
                #     other_scaled = other_scaled.loc[other_scaled['matches_available'] >= 1, :]
                current_score = base_scaled.loc[b_id, score_name]
                distances = np.abs(other_scaled[[score_name]] - current_score)
                tolerance_mask = distances[score_name] <= sd_tolerance
                available_mask = other_scaled['matches_available'] >= 1
                distances = distances.loc[available_mask & tolerance_mask, :]
                # distances = distances.loc[tolerance_mask, :]
                if len(distances) == 0:
                    continue

                if method == 'min':
                    distances = distances.sort_values(by = score_name).head(n_neighbors)
                elif method == 'random':
                    distances = distances.sample(min(n_neighbors, len(distances)))
                else:
                    pass

                candidates = distances.index
                other_scaled.loc[other_scaled.index.isin(candidates), 'matches_available'] -= 1

                for o_id in candidates:
                    match_id += 1
                    matches[match_id] = {base_name: b_id, other_name: o_id}
                    # matches[match_id][base_name] = b_id
                    # matches[match_id][other_name] = o_id
                    if base_name == 'treatment':
                        treatment_matches[b_id].append(match_id)
                        control_matches[o_id].append(match_id)
                    else:
                        treatment_matches[o_id].append(match_id)
                        control_matches[b_id].append(match_id)

        matched_data = pd.concat([base, other], axis = 0)
        matched_data[match_id_name] = matched_data.index
        # !note, that .get() do not set unexisting key in defaultdict while [] does !
        matched_data[match_id_name] = matched_data[match_id_name].apply(lambda x: control_matches.get(x, None) or treatment_matches.get(x, None))
        matched_data = matched_data.loc[matched_data[match_id_name].apply(lambda x: x is not None), :]
        matched_data = matched_data.explode(match_id_name).drop_duplicates([match_id_name, treatment_name]).sort_values(by = [match_id_name, treatment_name])

        result = {}
        result['matched_data'] = matched_data
        result['matches'] = dict(matches)
        result['treatment_matches'] = dict(treatment_matches)
        result['control_matches'] = dict(control_matches)

        return result

    def match_by_group(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        score_name: Union[str, List[str]],
        group_name: str,
        **kwargs
    ):
        """
        same as match, but makes procedure separately for groups by `group_name`
        """
        matched = data.groupby(group_name).apply(
            lambda x: self.match(data=x, treatment_name=treatment_name, score_name=score_name, **kwargs)['matched_data']
        )
        return {'matched_data': matched.reset_index(level=0, drop=True), 'group_name': group_name}

    def compare(
        self,
        initial_data: pd.DataFrame,
        matched_data: pd.DataFrame,
        treatment_name: str,
        score_name: str,
        features: List[str] = None,
        n_iterations: int = 100,
        seed: int = None,
    ):
        """
        so far doesn't work with categorical features
        """
        results = {}

        treatment_color = self.treatment_color
        control_color = self.control_color

        if not features:
            features = []

        t1_before = create_table_one(initial_data, treatment_col = treatment_name, features = [score_name] + features)
        t1_after = create_table_one(matched_data, treatment_col = treatment_name, features = [score_name] + features)
        t1_compare = pd.merge(t1_before, t1_after, left_index = True, right_index = True, how = 'outer', suffixes = ('_before', '_after'))
        # display(t1_compare)
        results['tableone'] = t1_compare

        fig, axes = plt.subplots(1, 2, figsize = (12, 5))
        plt.close(fig)
        _ = plot_metric_distribution_by_classes(
            initial_data, class_name = treatment_name, metric_name = score_name,
            class_label_mapping = {1: 'treatment', 0: 'control'},
            ax = axes[0], title = 'Before Matching'
        )
        _ = plot_metric_distribution_by_classes(
            matched_data, class_name = treatment_name, metric_name = score_name,
            class_label_mapping = {1: 'treatment', 0: 'control'},
            ax = axes[1], title = 'After Matching'
        )
        # display(fig)
        results['density'] = fig

        tb, cb = initial_data.loc[initial_data[treatment_name] == 1, score_name], initial_data.loc[initial_data[treatment_name] == 0, score_name]
        ta, ca = matched_data.loc[matched_data[treatment_name] == 1, score_name], matched_data.loc[matched_data[treatment_name] == 0, score_name]

        ecdf_t_before = EmpiricalDistribution(tb)
        ecdf_c_before = EmpiricalDistribution(cb)
        ecdf_t_after = EmpiricalDistribution(ta)
        ecdf_c_after = EmpiricalDistribution(ca)

        ks_p_b = ks_2sample_bootstrap(tb, cb, n_iterations = n_iterations, seed = seed)
        ks_p_a = ks_2sample_bootstrap(ta, ca, n_iterations = n_iterations, seed = seed)

        fig, axes = plt.subplots(1, 2, figsize = (12, 5))
        plt.close(fig)
        ecdf_c_before.plot_cdf(ax = axes[0], label = 'control', color = control_color)
        ecdf_t_before.plot_cdf(ax = axes[0], label = 'treatment', color = treatment_color, xlabel = score_name, title = f'Before Matching: KS-pvalue: {ks_p_b:.4f}')
        ecdf_c_after.plot_cdf(ax = axes[1], label = 'control', color = control_color)
        ecdf_t_after.plot_cdf(ax = axes[1], label = 'treatment', color = treatment_color, xlabel = score_name, title = f'After Matching: KS-pvalue: {ks_p_a:.4f}')
        results['ecdf'] = fig

        return results
