import copy
import itertools
import logging
import re
from collections import namedtuple
from collections.abc import Sized
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Union

import cobra
import cobra.sampling
import numpy as np
import pandas as pd
from cobra import Model
from cobra.sampling import OptGPSampler, ACHRSampler
from cobra.exceptions import OptimizationError
from cobra.io import load_model
from cobra.sampling import sample
from cobra.util import interface_to_str
from joblib import Parallel, delayed
from micom.community import Community
from micom.logger import logger
from micom.solution import (
    crossover,
    optimize_with_fraction,
    optimize_with_retry,
    reset_solver,
    solve,
)
from micom.util import (
    COMPARTMENT_RE,
    _apply_min_growth,
    _format_min_growth,
    add_var_from_expression,
    adjust_solver_config,
    check_modification,
    clean_ids,
    compartment_id,
    get_context,
    reset_min_community_growth,
)
from optlang.interface import OPTIMAL
from optlang.symbolics import Zero
from rich.progress import track
from scipy.stats import kruskal, rankdata
from scipy.stats.mstats import gmean
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .community_model import (
    HostComModel,
    _sample_across_conditions,
    set_community_limits,
)


logger = logging.getLogger("logger")


# how many bootstrapping reps to use for computing z-scores
Z_SCORE_BOOTSTRAP = 10000


## Initial functions
def get_metab_biomass_contribs(model):
    """Get the stoichiometric contribution of each metabolite
    towards the biomass function of the model"""
    biomass_rxn = model.reactions.query(lambda x: x.id.lower().startswith("biomass"))[0]
    biomass_metab = [str(m) for m in biomass_rxn.metabolites]

    # get the contribution coefficient of each metabolite towards growth
    metab_biomass_contrib = {
        m: -coeff
        for m, coeff in zip(biomass_metab, biomass_rxn.get_coefficients(biomass_metab))
    }

    return metab_biomass_contrib


def rxn_to_contrib(contribs, rxn):
    """Given a reaction and the stoichiometric coefficients of
    the metabolites towards biomass (negated) get the  mmol/gDW of biomass
    flux for each mmol/gDW of this reaction."""

    coeffs = dict(zip(rxn.metabolites, rxn.get_coefficients(rxn.metabolites)))
    return sum(contribs.get(str(metab), 0) * coeff for metab, coeff in coeffs.items())


def get_rxn_contribs(model):
    """Get the contribution coefficient of every reaction towards
    the biomass reaction. a coeff of X means that X unit of flux
    through that reaction translates to X units of flux towards the BM reaction.
    """
    metab_biomass_contrib = get_metab_biomass_contribs(model)
    return {
        rxn.id: rxn_to_contrib(metab_biomass_contrib, rxn) for rxn in model.reactions
    }


def get_bm_contrib_samples(model, n_samples=1000):
    """Using flux samples, return for all the reactions across all samples,
    the amount of flux that reaction is contributing towards positive flux through the biomass
    reaction of the model.

    The resulting table can be used to compute the statistics of these distributions
    and identify reactions to which to add reactions for "growth-coupling".
    """

    # get the samples
    flux_sample = cobra.sampling.sample(model, n_samples)

    # pivot the table
    long_flux_df = flux_sample.melt(
        value_vars=list(flux_sample.columns), value_name="flux", var_name="reaction"
    )

    # get the contributions
    contrib_df = (
        pd.DataFrame.from_dict(
            get_rxn_contribs(model), orient="index", columns=["coeff"]
        )
        .reset_index()
        .rename(columns={"index": "reaction"})
    )

    # add the coefficients and compute the total bm contribution
    merged_df = pd.merge(long_flux_df, contrib_df, on="reaction")
    merged_df.loc[:, "bm_contrib"] = merged_df["flux"] * merged_df["coeff"]

    return merged_df


## EXPERIMENTAL DESIGNER CODE


def get_flux_samples(
    mod, n_samples=1000, n_proc=1, n_batches: int = 10, thinning: int = 100, margin:float = 0.1,
    min_growth:Optional[float] = None
):
    """Using flux samples, return for all the reactions across all samples,
    the amount of flux that reaction is contributing towards positive flux through the biomass
    reaction of the model.

    The resulting table can be used to compute the statistics of these distributions
    and identify reactions to which to add reactions for "growth-coupling".
    """

    logger.info(f'plain sampling of model (no solve) with n_samples={n_samples}, n_batches={n_batches}, thinning={thinning}, min_growth={min_growth}, n_proc={n_proc}')
    # use context to revert destructive seting of limits
    with mod as model:
        if min_growth is not None:
            # apply minimum growth constraint if necessary
            _apply_min_growth(mod, _format_min_growth(min_growth, mod.taxa))  
        # get the samples
        if n_proc == 1:
            flux_sample = cobra.sampling.sample(
                model, n_samples, thinning=thinning, method="achr"
            )
        else:
            # optgp tends to be more performant only n_samples > 1000
            # we use batchign for better results
            sampler = OptGPSampler(model, processes=n_proc, thinning=thinning)
            flux_sample = sampler.batch(n_samples // n_batches, n_batches)
            flux_sample = pd.concat(list(flux_sample), axis="rows")

        # pivot the table
        long_flux_df = flux_sample.melt(
            value_vars=list(flux_sample.columns), value_name="flux", var_name="reaction"
        )
        long_flux_df["samp"] = long_flux_df.groupby("reaction").cumcount()

        return long_flux_df


def flux_range_to_cond(flux_df: pd.DataFrame, model: Optional[cobra.Model] = None):
    """Given a DF as obtained from `get_potential_factors`
    return a dictionary of experimental factors. If model is present
    will use the true upper bound"""

    # convert to records
    data_list = (
        flux_df.reset_index()
        .rename(columns={"index": "reaction"})
        .to_dict(orient="records")
    )

    # intialize reactions
    exp_conds = {rxn: {} for rxn in list(flux_df.reaction) + ["BASELINE"]}

    for row in data_list:
        # create a condition name
        # cond_name = re.sub(r"^EX_(.*)_e$", r"\1-", row["reaction"])  # TODO: see if it affects anything
        cond_name = row["reaction"]  # just the reaction name
        ub = 1e3  # normal upper bound
        if model is not None:
            # get it from the model if possible
            ub = model.reaction.get_by_id(row["reaction"]).ub

        exp_conds[cond_name][cond_name] = (
            max(row["min"], row["max"]) + 1e-5,
            ub,
        )  # and the lower part to the rest
        for other_cond in exp_conds.keys():
            # add the upper bound of the interval to baseline and other factors
            if other_cond != cond_name:
                exp_conds[other_cond][cond_name] = (
                    min(row["min"], row["max"]) + 1e-5,
                    ub,
                )

    return exp_conds


def set_exchanges(model, bounds):
    """Given a model set the specified in the dictionary
    The reactions are matched in a fuzzy way.

    The identifiers can be passed as the full reaction id or the metabolite id.
    """
    for rxn_pattern, (lb, ub) in bounds.items():
        matched_ex = model.exchanges.query(
            lambda x: (rxn_pattern.lower() == x.id.lower())
        )
        if len(matched_ex) == 0:
            # no exact reaction match, do fuzzy match
            matched_ex = model.exchanges.query(
                lambda x: (rxn_pattern.lower() in x.id.lower())
            )

        if len(matched_ex) > 1:
            logger.warning(
                f"{len(matched_ex)} matched for {rxn_pattern}. picking first one"
            )
        if len(matched_ex) == 0:
            logger.error(f"no exchange matched {rxn_pattern}")
            raise ValueError(f"no exchange matched {rxn_pattern}")

        # get the first one
        matched_ex = matched_ex[0]
        logging.info(f"setting the bounds of {matched_ex.id} to ({lb}, {ub})")
        matched_ex.bounds = (lb, ub)

    return model


def get_exchange_bounds_df(model):
    """Get the exchange boudns of a model"""
    ex_dict = [
        {
            "ex_id": ex.id,
            "lb": ex.bounds[0],
            "ub": ex.bounds[1],
        }
        for ex in model.exchanges
    ]
    return pd.DataFrame.from_records(ex_dict)


def df_for_metab(samps, metab):
    """Takes a metabolite and a sample DF and returns
    the production fluxes for each sample for that metabolite"""

    # get production coeffs
    rxn_coeffs = [
        {"reaction": rxn.id, "coeff": rxn.get_coefficient(metab)}
        for rxn in metab.reactions
    ]
    rxn_coeffs_df = pd.DataFrame.from_records(rxn_coeffs)

    # merge coeffs and compute flux
    result = pd.merge(samps, rxn_coeffs_df, on="reaction")
    result["flux"] = result["flux"] * result["coeff"]
    # keep only prod fluxes
    result = result.loc[(result["coeff"] > 0) & (result["flux"] > 0), :]

    # add total fluxes
    result = result.groupby(["samp", "exp_cond"]).agg({"flux": "sum"})
    result["metabolite"] = metab.id
    return result.reset_index()


def get_metab_fluxes(samps, model):
    """Given a model and samples for its fluxes return a dataframe
    with the values for the production fluxes of its cytosolic metabolites"""

    # get cytosolic metabs
    cyt_metabs = model.metabolites.query(lambda x: "c" == x.compartment)
    results = [df_for_metab(samps, metab) for metab in cyt_metabs]

    # return concatenated results
    return pd.concat(results, axis="rows")


def compute_condition_stats(
    results, alpha: float = 0.05, correction: str = "bonferroni", by: str = "reaction", pre_scale: bool = True
):
    """Compute the differential expression of all ractions across all
    combinations of experimental results.

    Compute log FC of absolute fluxes as well as the p values of
    groups of fluxes.
    """

    def compute_stats(x):
        if np.all(x["norm_flux_x"] == x["norm_flux_y"]):
            p_val = 1  # identical sets
        else:
            h_stat, p_val = kruskal(np.abs(x["norm_flux_x"]), np.abs(x["norm_flux_y"]))

        # the z score is computed with bootstrapping over the real samples
        samp_x = x.loc[:, "norm_flux_x"].sample(n=Z_SCORE_BOOTSTRAP, replace=True)
        samp_y = x.loc[:, "norm_flux_y"].sample(n=Z_SCORE_BOOTSTRAP, replace=True)

        # mean_diff / (std_diff / sqrt(n))
        z_score = (
            np.mean(samp_y - samp_x)
            / (np.std(samp_y - samp_x)  + 1e-15)
            * np.sqrt(Z_SCORE_BOOTSTRAP)
        )

        # add pseudo"count" at the end to avoid 0 value
        mean_l2_abs_fc = np.log2(
            (np.abs(np.mean(x["flux_y"])) + 1e-17)
            / (np.abs(np.mean(x["flux_x"])) + 1e-17)
        )
        return pd.Series(
            {
                "p": p_val,
                "log2_fc": mean_l2_abs_fc,
                "z_score": z_score,
                "mean_flux1": np.mean(x["flux_x"]),
                "mean_flux2": np.mean(x["flux_y"]),
                "mean_abs_flux1": np.mean(np.abs(x["flux_x"])),
                "mean_abs_flux2": np.mean(np.abs(x["flux_y"])),
            }
        )
    
    # ensure consistency of order
    unique_conds = set(results["exp_cond"].unique())
    exp_cond_values = ["BASELINE"] + sorted(unique_conds - {"BASELINE"})

    combinations = list(itertools.combinations(exp_cond_values, 2))
    dfs = []  # results here

    # scale if possible flag is set
    if pre_scale:
        results = results.groupby(by).apply(
            lambda x: x.assign(norm_flux=(x.flux - x.flux.mean()) / (x.flux.std() + 1e-16))
        ).reset_index(drop=True)
    else:
        results = results.assign(norm_flux = results.flux)

    for combination in combinations:
        fst_cond_df = results.loc[results["exp_cond"] == combination[0]]
        snd_cond_df = results.loc[results["exp_cond"] == combination[1]]
        merged_df = pd.merge(fst_cond_df, snd_cond_df, on=[by, "samp"])
        res_df = merged_df.groupby(by).apply(compute_stats)

        # add multiple testing correction
        res_df["p_adj"] = multipletests(res_df["p"], alpha=alpha, method=correction)[1]

        # add descriptors
        res_df["cond1"] = combination[0]
        res_df["cond2"] = combination[1]
        dfs.append(res_df)

    # some lines may be NaN drop them
    return pd.concat(dfs).dropna()


# def flux_sample_experiment(
#     model: cobra.Model, exp_conds: Dict[str, float], n_samples: int = 100
# ):
#     """Given a dictionary of experimenta cond name -> bounds as expected by `set_exchanges`
#     return a design matrix with the excahnge fluxes for those conditions as well as a concatenated table as the results for all experimental conditions as computed by `compute_bm_contrib_stats`.
#     """

#     # dataframes of experimental conditions
#     exp_conditions_dfs = []
#     results_dfs = []

#     for exp_cond_name, bound_dict in exp_conds.items():
#         model_cpy = copy.deepcopy(model) # copy the model
#         set_exchanges(model_cpy, bound_dict)  # set the bounds

#         # add the dict to the experimental condition dict
#         exp_cond_df = get_exchange_bounds_df(model_cpy)
#         exp_cond_df["exp_cond"] = exp_cond_name
#         exp_conditions_dfs.append(exp_cond_df)  # append it to the list

#         # run the experiment
#         results_df = get_bm_contrib_samples(model_cpy, n_samples)
#         results_df["exp_cond"] = exp_cond_name
#         results_dfs.append(results_df)

#     return {
#         "design": pd.concat(exp_conditions_dfs, axis="rows"),
#         "results": pd.concat(results_dfs, axis="rows"),
#     }


def get_potential_factors(
    model: cobra.Model,
    samp: Optional[pd.DataFrame] = None,
    n_samples: int = 100,
    n_proc: int = 1,
    n_batches: int = 1,
    thinning: int = 100,
    use_quantiles: bool = True,
    min_growth: float = 1e-5
):
    """Given a model, run flux sampling and return
    the exchanges with negative median and their min - max ranges."""
    if samp is None:
        samp = get_flux_samples(
            model,
            n_samples=n_samples,
            n_proc=n_proc,
            n_batches=n_batches,
            thinning=thinning,
            min_growth=min_growth
        )

    # get the exchange reactions
    exchanges = [rxn.id for rxn in model.exchanges]
    # get the fluxes for the exchange reaction
    samp = samp.loc[samp["reaction"].isin(exchanges), ["reaction", "flux"]]
    samp["samp"] = samp.groupby("reaction").cumcount()  # add sample number

    # do do some descriptive statistics for all exchanges
    res = samp.groupby("reaction").agg(
        min=pd.NamedAgg(column="flux", aggfunc="min"),
        median=pd.NamedAgg(column="flux", aggfunc="median"),
        mean=pd.NamedAgg(column="flux", aggfunc="mean"),
        max=pd.NamedAgg(column="flux", aggfunc="max"),
        q5=pd.NamedAgg(column="flux", aggfunc=lambda x: np.quantile(x, 0.05)),
        q95=pd.NamedAgg(column="flux", aggfunc=lambda x: np.quantile(x, 0.95)),
    )

    # if we want to use quantiles, use to 5 and bottom 5 for min and max
    if use_quantiles:
        res["min"] = np.minimum(res["q95"], res["q5"])
        res["max"] = np.maximum(res["q95"], res["q5"])
    else:
        res["min"], res["max"] = np.minimum(res["min"], res["max"]), np.maximum(res["min"], res["max"])

    res['range'] = np.abs(res["min"] - res["max"])

    # filter to keep reactions that are import (negative flux by convention) and vary at least a bit
    # TODO: not sure if the negative import flux is for all (should handle it with some compartment checks)
    res = res.loc[
        (res["median"] <= 1e-16) & (res['range'] > 1e-6)
    ].sort_values("median")
    res = res.reset_index()

    # sumbsample sample
    samp = samp[samp.reaction.isin(res['reaction'])].copy()

    # Grouping and sorting the DataFrame
    grouped = samp.groupby('samp').sum().sort_values('flux')

    # Find the indices of groups with the highest and lowest sum values
    max_idx = int(grouped.idxmax())
    min_idx = int(grouped.idxmin())

    # Filter the dataframe to keep only the groups with the highest and lowest sum values
    s = samp[samp['samp'].isin([max_idx,])].reset_index()
    
    s = s.groupby("reaction").agg(
        # min=pd.NamedAgg(column="flux", aggfunc="min"),
        max=pd.NamedAgg(column="flux", aggfunc="max"),
    ).reset_index()

    merged_df = s.merge(res, on='reaction', how='left')
    s['median'] = merged_df['median']
    s['mean'] = merged_df['mean']
    s['min'] = merged_df['min']

    return s



def format_sample_to_conditions(samp):
    """
    Utility function for the experimental designer.

    Given a a results from `sample_across_conditions`, format it to be processable by
    the differentail analysis we use downstream."""

    # columns and corresponding conditions
    lvl_cols = {
        col.replace("_lvl", ""): col for col in samp.columns if col.endswith("_m_lvl")
    }
    # dfs to concat
    results_df = []

    # create baseline condition
    baseline_conds = samp.loc[
        np.all(samp.loc[:, lvl_cols.values()] == 1, axis=1)
    ].copy()
    baseline_conds["exp_cond"] = "BASELINE"
    baseline_conds["samp"] = baseline_conds.reset_index().index
    results_df.append(baseline_conds)

    # now the rest
    for cond_name, col in lvl_cols.items():
        # get the conditions
        current_cond = samp.loc[samp[col] == 0].copy()
        current_cond["exp_cond"] = cond_name
        current_cond["samp"] = current_cond.reset_index().index
        results_df.append(current_cond)

    return pd.concat(results_df, axis="rows").reset_index(drop=True)


def limit_combinations(
    limits: pd.DataFrame, conditions: Optional[Union[int, List[str]]], levels: int = 2
):
    """Given the limit data frame, iterate over all the combinations

    Returns a pair of (spec, df) where the spec cotains (reaction_name, level, lower_bound)
    The df is the actual spec to pass to the sampling function.
    """
    limits = limits.copy()  # copy because of destructive ops

    # use all conditions in the experimental designer
    # by default
    if conditions is None:
        conditions = limits["reaction"]
    if isinstance(conditions, int):
        # crop the first few ones if it's an int that was passed
        conditions = (limits.iloc[:conditions])["reaction"]

    # all binary combinations
    limits["reaction"] = limits["reaction"].str.replace("_e$", "_m", regex=True)

    for comb in itertools.product(
        np.linspace(0, 1, num=levels), repeat=len(conditions)
    ):
        limits["comb"] = 1.0  # all 1
        limits.loc[
            limits["reaction"].isin(conditions), "comb"
        ] = comb  # just the selected are varied
        # apply tolerance param so the lower limit isn't that small
        limits["lb"] = limits["min"] * limits["comb"] + limits["max"] * (
            1 - limits["comb"]
        )  # get the interpolation for the level

        # name = ', '.join(f'{rxn}{"+" if c == 0 else "-"}' for rxn, c in zip(limits['reaction'], comb))
        spec = [
            (rxn, comb, lb)
            for rxn, comb, lb in zip(limits["reaction"], limits["comb"], limits["lb"])
        ]

        yield spec, limits[["reaction", "lb"]]


class ExperimentalDesigner(object):
    """Experimental desginer class

    Allows us to identify potential experimental conditions and to
    select target reactions and metabolites that are differnetially active
    within and between the conditions for all the models, as well as generally
    high or low fluxes"""

    _models: Dict[str, Model]  # the models used for experimental design
    n_samples: int
    n_proc: int
    n_batches: int
    thinning: int
    progress: bool
    _exp_conds: list = []

    rxn_samples: pd.DataFrame = None
    metab_samples: pd.DataFrame = None

    def __init__(
        self,
        models: Union[Dict[str, Model], List[Model], HostComModel],
        n_samples: int = 1000,
        range_n_samples: int = 30,
        n_proc: int = 1,
        n_batches: int = 1,
        thinning: int = 100,
        condition_limit: int = None,
        progress: bool = True,
        save_samples: bool = False,
        min_growth: float = 1e-3,
        exp_factors: Optional[pd.DataFrame] = None,
        pre_scale: bool = False,
        margin: float = 0.1
    ):
        """Create the experimenta designer class. Accepts either a list of models or a full community model.
        If we use a list of models, we default to using the naive method of finding the feasible conditions with the
        overlapping limit intervals. This doesn't take into account possible interactions, such as competition for the
        same substrate.

        If `save_samples` is true, save all the samples for later inspection. Defaults to true.
        """

        # use the models from HostHostComModel
        # save the model if so
        self._com_model: Optional[HostComModel] = None
        if isinstance(models, HostComModel):
            self._com_model = models
            models = models.initial_models

        # create dict
        if isinstance(models, list):
            models = {(mod.name or mod.id): mod for mod in models}

        self._models: List[cobra.Model] = models
        self.n_samples: int = n_samples
        self.n_proc: int = n_proc
        self.n_batches: int = n_batches
        self.progress: bool = progress
        self.thinning: int = thinning
        self._save_samples: bool = save_samples
        self.condition_limit: Optional[int] = condition_limit
        self.range_n_samples: int = range_n_samples
        self.min_growth: float = min_growth
        self._exp_factors: Optional[pd.DataFrame] = None
        self.pre_scale: bool = pre_scale
        self._margin: float = margin

        if exp_factors is not None:
            logger.info('Experimental factors passed through param')
            self._exp_factors = pd.DataFrame(exp_factors)

        # run the computations
        self._compute_exp_factors()

    def _compute_exp_factors(self):
        # loop individual models
        # if we don't have a global community model
        if self._com_model is None:
            pass
            raise ValueError('Requires a community model. Individual models are not supported anymore.')

        # branch for community model
        else:
            # if exp factors is not passed
            # compute it on the fly
            if self._exp_factors is None:
                logger.info('Running simulation to compute experimental factors.')
                # get the potential factors from the global model
                self._exp_factors: pd.DataFrame = get_potential_factors(
                    self._com_model,
                    n_samples=self.range_n_samples,
                    n_proc=self.n_proc,
                    n_batches=self.n_batches,
                    thinning=self.thinning,
                    min_growth=self.min_growth
                )

            # replace _m with _e to make it so that it translate to the internal models
            self._exp_factors = self._exp_factors.reset_index()[
                ["reaction", "min", "max"]
            ]  # keep the essentials
            self._exp_factors["reaction"] = self._exp_factors["reaction"].str.replace(
                "_m$", "_e", regex=True
            )
            print("FROM COMMUNITY MODEL:")  # debug

        self._exp_factors["range"] = np.abs(
            self._exp_factors["max"] - self._exp_factors["min"]
        )
        self._exp_factors = self._exp_factors.sort_values(
            "range", ascending=False
        )  # sort by the range, from highest to lowest

        # default to all of them
        if self.condition_limit is None:
            self.condition_limit = len(self._exp_factors)

        # truncate em
        self._exp_factors = self._exp_factors.iloc[:self.condition_limit]
        print(self._exp_factors)  # debug

        self._rxn_diffs_results: Dict[str, pd.DataFrame] = dict()
        self._metab_diffs_results: Dict[str, pd.DataFrame] = dict()

        # conditional execution based on community or indiv models
        if self._com_model is not None:
            self._get_conditions_com_model()
        else:
            raise('Model must be a community model.')

    def _get_conditions_com_model(self):
        """If we have a community model get differential reactions per model using
        community wide sampling."""

        print("COMMUNITY MODEL SAMPLING")
        # fix reaction names. we want to change the external m
        factors = self._exp_factors.copy()
        factors["reaction"] = factors["reaction"].str.replace(r"_e$", "_m", regex=True)

        # get all binary combinations of limits
        # must be list, for reasons
        # keep only the baselien or those that are all 1 or just 1 == 1
        lim_combs = list(
            limit_combinations(
                self.exp_factors, conditions=self.condition_limit, levels=2
            )
        )

        
        # logging info
        logger.info(f'preparing to sample {self.n_samples} for the following specs: ' + '\n'.join(str(spec) for spec, _ in lim_combs))

        # sample with them
        full_res = _sample_across_conditions(
            self._com_model,
            lim_combs,
            n_samples=self.n_samples,
            n_proc=self.n_proc,
            # must be passed via the kwargs
            samp_kwargs=dict(
                n_batches=self.n_batches, thinning=self.thinning,
                margin=self._margin,  
                clamp_all=False  # debatable what to do here
            ),

            # apply min growht constraints
            sim_kwargs=dict(
                min_growth=self.min_growth
            )
        )

        # format the results so it makes sens
        full_results = format_sample_to_conditions(full_res)
        full_results = {"results": full_results, "design": {}}  # does not return it

        # go to the _e naming of the reactions to make them correspond to the model
        full_results["results"]["exp_cond"] = full_results["results"][
            "exp_cond"
        ].str.replace(r"_m$", "_e", regex=True)
        if self._save_samples:
            # save all the samples
            self.full_samples = full_results["results"]

        idx = tqdm(self._models.items()) if self.progress else self._models.items()
        # get the different conditions for different models
        for mod_name, mod in idx:
            # get the reactions that end with the current model's name
            all_exp_cond_results = {"design": full_results["design"]}
            all_exp_cond_results["results"] = full_results["results"].loc[
                full_results["results"]["reaction"].str.endswith(mod_name)
            ]
            all_exp_cond_results["results"] = all_exp_cond_results[
                "results"
            ].copy()  # copy for operation

            # drop the suffix
            all_exp_cond_results["results"]["reaction"] = all_exp_cond_results[
                "results"
            ]["reaction"].str.replace(rf"_*{mod_name}$", "", regex=True)

            # compute diff rxn results
            diff_stats_df = compute_condition_stats(
                all_exp_cond_results["results"], by="reaction", pre_scale=self.pre_scale
            )
            self._rxn_diffs_results[mod_name] = diff_stats_df.reset_index()

            # compute diff metab results
            metab_samps = get_metab_fluxes(all_exp_cond_results["results"], mod)

            diff_stats_df = compute_condition_stats(metab_samps, by="metabolite", pre_scale=self.pre_scale)
            self._metab_diffs_results[mod_name] = diff_stats_df.reset_index()

            # compute the differential stats for the biomasses
            self._org_diffs_results = pd.concat(
                diff.loc[diff.reaction.str.lower().str.startswith('biomass')].assign(organism = org).drop('reaction', axis='columns')
                for org, diff in self._rxn_diffs_results.items()
            )

    @staticmethod
    def _parse_specs(s):
        """Parse the ranking specifications to use"""
        regex = r"([DCHLAIPN])([\d,a-zA-Z0-9-_]*|(\([\d,a-zA-Z0-9-_]*\)))(?:\s+|$)"
        matches = re.findall(regex, s)

        result = []

        for match in matches:
            identifier = match[0]
            value = match[1]

            if identifier in {"D", "C", "A", "I", "P" ,"N"}:
                value = value.replace("(", "").replace(")", "")
                if "," in value:
                    value = tuple(value.split(","))
                else:
                    value = (value,)

            elif identifier.startswith("H"):
                value = value.replace("(", "").replace(")", "")
                if value == "B":
                    identifier = "H"
                    value = "B"
                else:
                    value = (value,)
            value = (int(val) if val.isdigit() else val for val in value)

            result.append((identifier, *value))

        return result

    def get_full_specs(self, spec_str: str):
        """Given a spec string in the format, return a list of tuples
        to be used with downstream tasks"""

        # parse the string
        results = self._parse_specs(spec_str)
        fin_results = []

        # for H, L, C and D append all conditions
        new_results = []
        for identifier, *values in results:
            if str(values[0]).strip() != "":
                new_results.append((identifier, *values))
            else:
                # 0 values
                if identifier in ["H", "L"]:
                    new_results += [
                        (identifier, cond) for cond in self.exp_conds
                    ]  # add all conditions
                if identifier in ["C", "D", "A", "I", "P", "N"]:
                    # add all combinations of conditions
                    new_results += [
                        (identifier, a, b)
                        for (a, b) in itertools.combinations(self.exp_conds, 2)
                    ]
        results = new_results  # with expanded ones

        for identifier, *values in results:
            new_vals = []

            if len(values) > 2:
                raise ValueError(f"Too many ids in {values}")

            # convert values to condition identifier
            for val in values:
                # convert to value
                if val == "B":
                    val = "BASELINE"  # B is shorthand for BASELINE
                elif isinstance(val, int):
                    val = self.exp_conds[val]  # index using val
                if not val in self.exp_conds:
                    raise ValueError(f"{val} not an existing exeperimental condition")
                new_vals.append(val)  # add to all vals

            if len(new_vals) == 1 and identifier not in ["H", "L"]:
                new_vals = ("BASELINE", new_vals[0])  # BASELINE by default

            # sort by the order the conditions show in
            new_vals = sorted(new_vals, key=lambda x: self.exp_conds.index(x))
            fin_results.append((identifier, *new_vals))  # append to list of all

        return list(set(fin_results))

    def get_target(
        self, spec: str, model: Optional[str] = NotImplemented, by: str = "reaction", diff_mode: str = "both", p_cutoff:float = 0.05
    ) -> pd.DataFrame:
        """Given a spec in the format:
        ` `
        Return a dataframe  with the ranks for each condition and a final rank with
        the rank product of th e other columns.

        diff_mode specifies whether to use z_scores or adjusted p values for the differentially
        active reactions or metabolites. Can be 'p', 'z' or 'both'
        """
        # verify differential activity mode.
        if diff_mode not in ["p", "z", "both"]:
            raise ValueError("`diff_mode` has to be in ['p', 'z', 'both']")
        if diff_mode == "both":
            diff_mode = ["p", "z"]
        else:
            diff_mode = [diff_mode]  # just keep p or z

        # check the value of the column
        if by not in ["reaction", "metabolite", "organism"]:
            raise ValueError('`by` must be either "reaction" or "metabolite" or "organism"')

        # check model name
        if by != 'organism' and model not in self._metab_diffs_results.keys():
            raise ValueError(
                f'model "{model}" must be in {list(self._metab_diffs_results.keys())}'
            )

        spec_list = self.get_full_specs(spec)

        # mapping fromspecification to column to rank
        id_to_col_map = {
            "H": ["mean_flux2"],
            "L": ["neg_mean_flux2"],
            # add p value or z score, or both
            "D": (["abs_z_score"] if "z" in diff_mode else [])
            + (["neg_p_adj"] if "p" in diff_mode else []),

            "P": ["positive1", "positive2"],
            "N": ["negative1", "negative2"],

            "C": (["neg_abs_z_score"] if "z" in diff_mode else [])
            + (["p_adj"] if "p" in diff_mode else []),

            "A": (["z_score"] if "z" in diff_mode else [])
            + (["neg_p_adj"] if "p" in diff_mode else []) + ['mean_flux2', 'mean_flux1'],

            "I": (["neg_z_score"] if "z" in diff_mode else [])
            + (["p_adj"] if "p" in diff_mode else []) + ['mean_flux2', 'mean_flux1'],

            # high or low BASELINE special cases
            "H*": ["mean_flux1"],
            "L*": ["neg_mean_flux1"],
        }
        # use either reactions or metabolites for this
        if by == "metabolite":
            data = self._metab_diffs_results[model]
        elif by == 'reaction': 
            data = self._rxn_diffs_results[model]
        elif by == 'organism':
            # organism
            data = self._org_diffs_results

        data = data.copy()  # copy so we don't modify the original one

        # do not use transfer reactions. drop them from the list of candidates
        if by == "reaction":
            # not ransfer reactions
            valid_rxns = [
                r.id
                for r in self._models[model].reactions.query(
                    lambda x: len(x.compartments) == 1
                    and not x.id.lower().startswith("ex_")
                )
            ]
            data = data.loc[
                data.reaction.isin(valid_rxns)
            ]  # drop tarnsfers and exports

        # create the columns used for selection downstream
        data.loc[:, "neg_mean_flux1"] = -data["mean_flux1"]
        data.loc[:, "neg_mean_flux2"] = -data["mean_flux2"]
        data.loc[:, "abs_z_score"] = np.abs(data["z_score"])
        data.loc[:, "neg_z_score"] = -data["z_score"]
        data.loc[:, "neg_abs_z_score"] = -data["abs_z_score"]
        data.loc[:, "neg_p_adj"] = -data["p_adj"]  # p_adj is used in some cases
        data.loc[:, "positive1"] = (data["mean_flux1"] > 0).astype(bool)
        data.loc[:, "positive2"] = (data["mean_flux2"] > 0).astype(bool)
        data.loc[:, "negative1"] = (data["mean_flux1"] < 0).astype(bool)
        data.loc[:, "negative2"] = (data["mean_flux2"] < 0).astype(bool)

        # get the columns and conditions to filter by
        patterns = [
            # C and D spec
            (
                {"spec": s[0], "col": id_to_col_map[s[0]], "cond1": s[1], "cond2": s[2]}
                if len(s) == 3
                else
                # H and L spec
                (
                    {
                        "spec": s[0],
                        "col": id_to_col_map[s[0]],
                        "cond1": self.exp_conds[0],
                        "cond2": s[1],
                    }
                    if s[1] != "BASELINE"
                    else
                    # special case for BASELINE
                    {
                        "spec": s[0],
                        "col": id_to_col_map[s[0] + "*"],
                        "cond1": s[1],
                        "cond2": self.exp_conds[-1],
                    }
                )
            )
            for s in spec_list
        ]

        # start with empty dataframe
        merged_df = None
        # iterate patterns
        for pat in patterns:
            # get the subset of rows for the conditions
            # NOTE: this works only if the conditions are in the order 'BASELINE' + sorted
            sub_df: pd.DataFrame = data.loc[
                ((data["cond1"] == pat["cond1"]) & (data["cond2"] == pat["cond2"])) | 
                ((data["cond1"] == pat["cond2"]) & (data["cond2"] == pat["cond1"])), :
            ]

            # iterate over all columns in the subset
            for src_col in pat["col"]:
                # get the colname to user with both the _raw and _rank suffixes
                new_col_name: str = (
                    f'{pat["spec"]}({pat["cond1"]},{pat["cond2"]})({src_col})'
                )

                # add the raw value and ranks
                sub_df.loc[:, new_col_name + "_raw"] = sub_df[src_col]
                if pat["spec"] in ['P', 'N']:
                    # flag for positivity
                    sub_df.loc[:, new_col_name + "_flag"] = sub_df[new_col_name + "_raw"]

                # flag for p threshold
                # and rank
                elif src_col == 'neg_p_adj' and pat["spec"] in ["D", "A", "I"]:
                    sub_df.loc[:, new_col_name + "_rank"] = rankdata(
                        sub_df[new_col_name + "_raw"]
                    )
                    # for D and A for example, higher rank means lower p_value
                    # we want low p values for that
                    sub_df.loc[:, new_col_name + "_flag"] = sub_df[new_col_name + "_raw"] > -p_cutoff

                elif src_col.startswith('mean') and pat["spec"] in ["D", "A", "I"]:
                    pass # no rank just keep the values so they are there

                elif src_col == 'p_adj' and pat["spec"] == "C":
                    sub_df.loc[:, new_col_name + "_rank"] = rankdata(
                        sub_df[new_col_name + "_raw"]
                    )
                    # positive, lowest p_values
                    sub_df.loc[:, new_col_name + "_flag"] = sub_df[new_col_name + "_raw"] > p_cutoff

                else:
                    sub_df.loc[:, new_col_name + "_rank"] = rankdata(
                        sub_df[new_col_name + "_raw"]
                    )

            # add the increase or decrease flag
            if pat["spec"] == 'A' or pat["spec"] == 'I':
                # get the col name for the fluxes
                col_mean1_name: str = 'mean_flux1'
                col_mean2_name: str = 'mean_flux2'

                col_name = f'{pat["spec"]}({pat["cond1"]},{pat["cond2"]})(differece)_flag'
                if pat['spec'] == 'A':
                    col_data = sub_df.loc[:, col_mean1_name] < sub_df.loc[:, col_mean2_name]
                else:
                    col_data = sub_df.loc[:, col_mean1_name] > sub_df.loc[:, col_mean2_name]

                sub_df.loc[:, col_name] = col_data


            # subset just the needed columns
            needed_cols = [
                col
                for col in sub_df.columns
                if col == by or col.endswith("_rank") or col.endswith("_raw") or col.endswith("_flag")
            ]
            sub_df = sub_df.loc[:, needed_cols]  # just keep the raw, rank and vy cols

            # initialize or merge
            if merged_df is None:
                merged_df = sub_df
            else:
                merged_df = pd.merge(merged_df, sub_df, on=by)

        # add the final rank prod
        rank_cols = [col for col in merged_df.columns if col.endswith("_rank")]
        flag_cols = [col for col in merged_df.columns if col.endswith("_flag")]
        
        bool_col = None
        for flag_col in flag_cols:
            if bool_col is None:
                bool_col = merged_df[flag_col]
            else:
                bool_col &= merged_df[flag_col]
        if bool_col is not None:
            merged_df['flag'] = bool_col.astype(bool)

        merged_df.loc[:, "rank_prod"] = gmean(
            merged_df.loc[:, rank_cols].values, axis=1
        )  # NOTE: no weights yet
        return merged_df

    @property
    def exp_conds(self) -> List[str]:
        """Returns all the experimental conditions used to generate the differential values.
        For example returns BASELINE and all the other reactions constrained.

        Returns:
            List[str]: The list of experimental conditions tested.
        """
        # get the names of conditions
        if len(self._exp_conds) == 0:
            # add baseline as a first one
            self._exp_conds = ["BASELINE"] + list(self._exp_factors["reaction"])
        return self._exp_conds

    # getter for models
    @property
    def models(self) -> Dict[str, Model]:
        return self._models

    @property
    def all_exp_factors(self) -> Dict[str, pd.DataFrame]:
        return self._all_exp_factors

    @property
    def exp_factors(self) -> pd.DataFrame:
        return self._exp_factors
