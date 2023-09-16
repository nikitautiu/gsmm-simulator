import copy
from collections import namedtuple
from collections.abc import Sized
from functools import partial
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cobra
import cobra.sampling
from cobra.sampling import OptGPSampler, ACHRSampler
import numpy as np
import pandas as pd
import joblib
from cobra.exceptions import OptimizationError
from cobra.io import load_model
from cobra.sampling import sample
from cobra.util import interface_to_str
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

from .utils import get_current_seed

logger = logging.getLogger("logger")


def optimize_with_retry_dual_sol(
    com, message: str = "could not get optimum."
) -> Tuple[float, float]:
    """Try to reset the solver and return both the community and host objective.
    Unlike the Micom variant, returns both the microbe community and host primals.
    """
    sol = com.optimize()
    host_growth: Optional[float] = None
    com_growth: Optional[float] = None

    if sol is None:
        reset_solver(com)
        sol = com.optimize()
    if sol is None:
        raise OptimizationError(message)
    else:
        return (
            com.variables.microbe_community_objective.primal,
            com.variables.host_objective.primal,
        )


def regularize_l2_norm(community, min_growth, add_host: bool = False):
    """Add an objective to find the most "egoistic" solution.

    This adds an optimization objective finding a solution that maintains a
    (sub-)optimal community growth rate but is the closest solution to the
    community members individual maximal growth rates. So it basically finds
    the best possible tradeoff between maximizing community growth and
    individual (egoistic) growth. Here the objective is given as the sum of
    squared differences between the individuals current and maximal growth
    rate. In the linear case squares are substituted by absolute values
    (Manhattan distance).

    Arguments
    ---------
    community : micom.Community
        The community to modify.
    min_growth : positive float
        The minimal community growth rate that has to be mantained.
    linear : boolean
        Whether to use a non-linear (sum of squares) or linear version of the
        cooperativity cost. If set to False requires a QP-capable solver.
    max_gcs : None or dict
        The precomputed maximum individual growth rates.

    """
    logger.info(f"adding L2 norm: {min_growth} (add_host={add_host})")
    l2 = Zero
    community.variables.community_objective.lb = min_growth
    context = get_context(community)
    if context is not None:
        context(partial(reset_min_community_growth, community))

    # only microbe community objectives
    for sp in community.microbe_taxa:
        taxa_obj = community.constraints["objective_" + sp]
        ex = sum(v for v in taxa_obj.variables if (v.ub - v.lb) > 1e-6)
        if not isinstance(ex, int):
            l2 += (community.scale * (ex**2)).expand()

    # update objective
    if add_host:
        community.objective = -l2 + community.variables.host_objective
        community.modification = "l2 regularization + host objective"
    else:
        community.objective = -l2
        community.modification = "l2 regularization"

    logger.info("finished adding tradeoff objective to %s" % community.id)


def cooperative_tradeoff(
    community, min_growth, fraction, host_fraction, fluxes, pfba, atol, rtol
):
    """Find the best tradeoff between community and individual growth."""
    with community as com:
        solver = interface_to_str(community.problem)
        check_modification(community)
        min_growth = _format_min_growth(min_growth, community.taxa)
        logger.info(f'applying minimum growth rate: {min_growth}')
        _apply_min_growth(community, min_growth)
 
        com.objective = com.scale * com.variables.community_objective
        min_growth_com, min_growth_host = optimize_with_retry_dual_sol(
            com, message="could not get community growth rate."
        )

        logger.info(f'tradeoff: solved initial optimization with: com={min_growth_com} host={min_growth_host}')
        # NOTE: These are part of the normal MICOM formulation, but we don't need them here
        # cause we don't actually want the values they control
        # min_growth_com /= com.scale
        # min_growth_host /= com.scale

        # print(min_growth_com, min_growth_host)

        if not isinstance(fraction, Sized):
            fraction = [fraction]
        else:
            fraction = np.sort(fraction)[::-1]

        if not isinstance(host_fraction, Sized):
            host_fraction = [host_fraction]
        else:
            host_fraction = np.sort(host_fraction)[::-1]

        # Add needed variables etc.
        regularize_l2_norm(com, 0.0)
        results = []
        for fr, h_fr in zip(fraction, host_fraction):
            logger.info(f'tradeoff: trying fraction microbe={fraction}, host={host_fraction}')

            com.variables.microbe_community_objective.lb = fr * min_growth_com
            com.variables.microbe_community_objective.ub = min_growth_com
            logger.info(f'tradeoff: set {com.variables.microbe_community_objective.name} lb and ub to {com.variables.microbe_community_objective.lb} and {com.variables.microbe_community_objective.ub}')

            com.variables.host_objective.lb = h_fr * min_growth_host
            com.variables.host_objective.ub = min_growth_host
            logger.info(f'tradeoff: set {com.variables.host_objective.name} lb and ub to {com.variables.host_objective.lb} and {com.variables.host_objective.ub}')

            sol = solve(community, fluxes=fluxes, pfba=pfba, atol=atol, rtol=rtol)
            logger.info(f'tradeoff: got solution {sol}')
            # OSQP is better with QPs then LPs
            # so it won't get better with the crossover
            if not pfba and sol.status != OPTIMAL and solver != "osqp":
                logger.info('tradeoff: solution not optimal, trying crossover')
                sol = crossover(com, sol, fluxes=fluxes)
                logger.info(f'tradeoff: got solution {sol}')
            results.append((fr, sol))
        if len(results) == 1:
            return results[0][1]
        return pd.DataFrame.from_records(results, columns=["tradeoff", "solution"])


class HostComModel(Community):
    initial_models: List[cobra.Model]
    model_weights: Mapping[str, float]
    max_exchange: float

    def __init__(
        self,
        models=None,
        abundances=None,
        microbe_taxa=None,
        id=None,
        name=None,
        rel_threshold=1e-6,
        mass=1,
        max_exchange=1000,
        solver=None,
        progress=True,
    ):
        super(Community, self).__init__(id, name)
        self.max_exchange = max_exchange

        # initialize models to nothing
        if models is None:
            models = dict()
        self.initial_models = models

        # set abundances to uniform
        if abundances is None:
            abundances = {m.name: 1.0 / len(models) for m in models}
        self.model_weights = abundances

        # set microbes to all
        if microbe_taxa is None:
            microbe_taxa = list(models.keys())

        # start building the model
        logger.info("building new micom model {}.".format(id))
        if not solver:
            solver = [
                s
                for s in ["cplex", "gurobi", "osqp", "glpk"]
                if s in cobra.util.solver.solvers
            ][0]
        logger.info("using the %s solver." % solver)
        if solver == "glpk":
            logger.warning(
                "No QP solver found, will use GLPK. A lot of functionality "
                "in MICOM will require a QP solver :/"
            )

        # solver specific tweaks
        self.solver.configuration.lp_method = "auto"
        self.solver.configuration.qp_method = "auto"
        self.solver.configuration.presolve = False
        self.solver = solver
        self._rtol = rel_threshold
        self._modification = None
        self.mass = mass
        self.__db_metrics = None
        adjust_solver_config(self.solver)

        # create taxonomy
        if len(models) > 0:
            taxonomy = pd.DataFrame.from_dict(
                data={"id": {k: k for k in models.keys()}, "abundance": abundances}
            )
        else:
            taxonomy = pd.DataFrame(data={"id": [], "abundance": []})
            taxonomy["id"] = taxonomy["id"].astype(str)
            taxonomy["abundance"] = taxonomy["abundance"].astype(np.float32)

        if taxonomy.id.str.contains(r"[^A-Za-z0-9_]", regex=True).any():
            logger.warning(
                "Taxa IDs contain prohibited characters and will be reformatted."
            )
            taxonomy.id = taxonomy.id.replace(r"[^A-Za-z0-9_\s]+", "_", regex=True)

        self.__taxonomy = taxonomy
        self.__taxonomy.index = self.__taxonomy.id
        self._Community__taxonomy = self.__taxonomy

        obj = Zero
        self.taxa = []
        index = list(models.keys())
        index = track(index, description="Building") if progress else index

        for idx in index:
            model = copy.deepcopy(models[idx])
            abund = abundances[idx]

            suffix = "__" + idx.replace(" ", "_").strip()
            logger.info("converting IDs for {}".format(idx))
            external = cobra.medium.find_external_compartment(model)
            logger.info(
                "Identified %s as the external compartment for %s. "
                "If that is wrong you may be in trouble..." % (external, idx)
            )
            for r in model.reactions:
                r.global_id = clean_ids(r.id)
                r.id = r.global_id + suffix
                r.community_id = idx
                # avoids https://github.com/opencobra/cobrapy/issues/926
                r._compartments = None
                # SBO terms may not be maintained
                if "sbo" in r.annotation:
                    del r.annotation["sbo"]
            for m in model.metabolites:
                m.global_id = clean_ids(m.id)
                m.id = m.global_id + suffix
                m.compartment += suffix
                m.community_id = idx
            logger.info("adding reactions for {} to community".format(idx))
            self.add_reactions(model.reactions)
            o = self.solver.interface.Objective.clone(
                model.objective, model=self.solver
            )
            obj += o.expression * abund
            self.taxa.append(idx)
            taxa_obj = self.problem.Constraint(
                o.expression, name="objective_" + idx, lb=0.0
            )
            self.add_cons_vars([taxa_obj])

            # patch value passing
            # create an abundace object as expected by the parent method
            RowObj = namedtuple("Info", ["abundance"])
            row = RowObj(abund)  # set to actual abundance

            # unmangle the parent private method name (ugly)
            self._Community__add_exchanges(
                model.reactions,
                row,
                external_compartment=external,
                internal_exchange=max_exchange,
            )
            self.solver.update()  # to avoid dangling refs

        # add the community objective variable
        com_obj = add_var_from_expression(self, "community_objective", obj, lb=0)
        self.objective = self.problem.Objective(com_obj, direction="max")

        # add additional data
        self.microbe_taxa = microbe_taxa

        # add microbe communinty objective
        total_obj = Zero
        for tax in self.microbe_taxa:
            taxa_obj = self.constraints["objective_" + tax]
            total_obj += taxa_obj.expression * self.abundances[tax]

        mic_com_obj = add_var_from_expression(
            self, "microbe_community_objective", total_obj, lb=0
        )

        # add host objective
        # consider everything that's not a microbe as host
        total_obj = Zero
        for tax in set(self.taxa) - set(self.microbe_taxa):
            taxa_obj = self.constraints["objective_" + tax]
            total_obj += taxa_obj.expression * self.abundances[tax]

        host_obj = add_var_from_expression(self, "host_objective", total_obj, lb=0)

    def cooperative_tradeoff(
        self,
        min_growth: float = 0.0,
        fraction: float = 1.0,
        host_fraction: float = 1.0,
        fluxes: bool = False,
        pfba: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        """Find the best tradeoff between community and individual growth.
        Finds the set of growth rates which maintian a particular community
        growth and spread up growth across all taxa as much as possible.
        This is done by minimizing the L2 norm of the growth rates with a
        minimal community growth.
        Parameters
        ----------
        min_growth : float or array-like, optional
            The minimal growth rate required for each individual. May be a
            single value or an array-like object with the same length as there
            are individuals.
        fraction : float or list of floats in [0, 1]
            The minum percentage of the microbial growth rate that has to be
            maintained. For instance 0.9 means maintain 90% of the maximal
            community growth rate. Defaults to 100%.
        host_fraction : float or list of floats in [0, 1]
            The minum percentage of the host growth rate that has to be
            maintained. For instance 0.9 means maintain 90% of the maximal
            host growth rate. Defaults to 100%.
        fluxes : boolean, optional
            Whether to return the fluxes as well.
        pfba : boolean, optional
            Whether to obtain fluxes by parsimonious FBA rather than
            "classical" FBA. This is highly recommended.
        atol : float
            Absolute tolerance for the growth rates. If None will use the solver
            tolerance.
        rtol : float
            Relative tolerqance for the growth rates. If None will use the
            solver tolerance.
        Returns
        -------
        micom.CommunitySolution or pd.Series of solutions
            The solution of the optimization. If fluxes==False will only
            contain the objective value, community growth rate and individual
            growth rates. If more than one fraction value is given will return
            a pandas Series of solutions with the fractions as indices.
        """
        if atol is None:
            atol = self.solver.configuration.tolerances.feasibility
        if rtol is None:
            rtol = self.solver.configuration.tolerances.feasibility

        return cooperative_tradeoff(
            self, min_growth, fraction, host_fraction, fluxes, pfba, atol, rtol
        )

    def sample_with_growth_margin(
        self,
        growth_rates: Optional[Dict[str, float]],
        clamp_all: bool = False,
        n_samples: int = 100,
        margin: float = 0.1,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        thinning: int = 100,
        n_proc: int = 1,
        n_batches: int = 1,
        min_growth: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """Given a set of growth rates from a solution, return a sample
        of all the fluxes when the growth rate is capped withing a margin of those values.
        """

        margin = max(margin, 1e-4)
        if margin > 1 and growth_rates is not None:
            logging.warning('margin set to > 1 and growth rates was passed')

        # if seed is none, replace with global numpy seed
        if seed is None:
            seed = get_current_seed()

        # select the type of clamping to do
        clamp_func = _clamp_com_host_growth
        if clamp_all:
            # can be clamping all indiv growth rates
            logger.info("clamping all growths")
            clamp_func = _clamp_growth
        else:
            # or just the host and communit growth
            logger.info("clamping community and host only")

        # use contxt for clamping
        with self as mod:
            # add positivity constraints to the growth rates
            _apply_min_growth(mod, _format_min_growth(min_growth, mod.taxa)) 
            if growth_rates is not None: 
                clamp_func(
                    mod, growth_rates, rel_margin=margin, atol=atol, rtol=rtol
                )  # use whichever constraint we selected - if we desire a margin

            # initialize sampler
            sampler = OptGPSampler(self, processes=n_proc, thinning=thinning, seed=seed)
            samps = sampler.batch(
                n_samples // n_batches, n_batches
            )  # may not divide ok. TODO: add warning
            samp = pd.concat(list(samps), axis="rows")  # concatenate the results

            return samp


def set_community_limits(
    com: HostComModel,
    limits: Union[Mapping[str, float], pd.DataFrame],
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    """Set the lower bounds to the given reactions using a context manager so
    they are reversible"""
    if isinstance(limits, pd.DataFrame):
        limits = dict(limits.set_index("reaction")["lb"])

    context = get_context(com)

    def reset(reaction, lb, ub):
        logger.info(f"resetting constraint for {reaction} to ({lb}, {ub})")
        com.reactions.get_by_id(reaction).bounds = (lb, ub)

    for reaction, lb in limits.items():
        rxn = com.reactions.get_by_id(reaction)  # get reatction with that id

        # set lower bound and add a resetter
        if context:
            context(partial(reset, reaction, *rxn.bounds))
        logger.info(f"setting constraint for {reaction} to lb={lb} ub={rxn.bounds[1]}")
        rxn.bounds = ((1.0 - rtol) * lb - atol, rxn.bounds[1])


def _process_chunk(chunk, model, default_sim_kwargs, defaults_samp_kwargs, n_samples, n_proc):
    """Given a chunk of limit specs and limit dataframes(list),
    process it and return the list of dataframes with the samples in the long format

    Used by the parallel function
    """

    # list of all samples collected from the ch unk
    all_samps: List[pd.DataFrame] = []

    for lim_spec, lims in chunk:
        # copy the model just to be sure
        # com = copy.deepcopy(model)
        com = copy.deepcopy(model)
        with com as mod:
            print("Started:", lim_spec)  # debug
            set_community_limits(mod, lims)  # set the limits of the exchange fluxes

            try:
                sol = None
                if defaults_samp_kwargs.get('margin', 0.1) > 1:
                    logger.info('Margin > 1, using sampling-only algorithm')
                    # solve the cooperative tradeoff
                    sol = mod.cooperative_tradeoff(fluxes=False, **default_sim_kwargs)
                    logging.info(f'when solving host-community for sampling got solution {sol.members["growth_rate"]}')  # debug

                # delegate the sampling to the community model
                samp = mod.sample_with_growth_margin(
                    sol.members["growth_rate"] if sol is not None else None,  # pass None if margin > 1
                    n_samples=n_samples,
                    n_proc=n_proc,
                    **defaults_samp_kwargs,
                )
                # print('Sampled:', lims)  # debug
            except Exception as err:
                raise ValueError(
                    f"Exception occured when sampling for limits {lim_spec}"
                ) from err

            flux_sample = samp
            # assign sample numbers to the flux samples in the long form
            long_flux_df = flux_sample.melt(
                value_vars=list(flux_sample.columns),
                value_name="flux",
                var_name="reaction",
            )
            long_flux_df["samp"] = long_flux_df.groupby("reaction").cumcount()

            # add columns that specify the level of each condition
            # unpack the values in (reaction, level (0-1), actual value)
            for rxn, lvl, _ in lim_spec:
                long_flux_df[f"{rxn}_lvl"] = lvl

        all_samps.append(long_flux_df)  # append it to the list of dataframes
    return all_samps


def _sample_across_conditions(
    com: HostComModel,
    limit_specs_and_lims: List[Any],
    n_samples: int = 1000,
    n_proc: int = 1,
    sim_kwargs: Dict[str, Any] = {},
    samp_kwargs: Dict[str, Any] = {},
):
    """Sample a community model across different levels of a given experimental factor"""
    # default args for the simulations and sampling function of the community
    default_sim_kwargs = dict(
        fraction=0.9,   # microbe fraction
        host_fraction=0.9,   # host fraction

        # prevents organisms from commiting altruistic suicide 
        # in the first step of the cooperative tradeoff
        min_growth=1e-5   
    )  # 75% 75% limits for the host and community growths
    defaults_samp_kwargs = dict(
        clamp_all=False, margin=0.1, thinning=100, n_batches=1
    )  # +-10% and clamp only host and total microbe rates (not all the individual ones)

    logger.info('preparing sampling for host community model')

    # update kwargs with the supplied params
    default_sim_kwargs.update(sim_kwargs)
    defaults_samp_kwargs.update(samp_kwargs)

    # choose how many procs we use in the outer vs inner loop
    # for higher thiining/number of samples, it is better to allocate 
    # to the sampler
    total_inner_samps: int = defaults_samp_kwargs['thinning'] * n_samples
    inner_proc: int = 1
    if total_inner_samps >= 1000:
        inner_proc = n_proc
        n_proc = 1

    # TODO swap for logging
    print(f'proc={n_proc} inner_proc={inner_proc} total_inner_samps={total_inner_samps}')

    # use joblib to parallelize the loop
    chunk_size = max(1, len(limit_specs_and_lims) // n_proc)  # create n_proc chunks

    # create the chunks of conecutive values
    chunks = [
        limit_specs_and_lims[i : i + chunk_size]
        for i in range(0, len(limit_specs_and_lims), chunk_size)
    ]
    all_samps = joblib.Parallel(n_jobs=n_proc)(
        joblib.delayed(_process_chunk)(
            chunk, copy.deepcopy(com), default_sim_kwargs, defaults_samp_kwargs, n_samples, inner_proc
        )
        for chunk in chunks
    )

    # flatten the list
    all_samps = [item for sublist in all_samps for item in sublist]

    # concatenate the samples
    raw_data = pd.concat(all_samps)

    return raw_data


def _clamp_com_host_growth(
    com: HostComModel,
    growth,
    rel_margin: float = 0.1,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    """Clampt the host and community objectives to within +- rel_margin of the optimals.
    Unlike the other growth clamping function, this allows the individual rates to vary.
    """
    context = get_context(com)

    def reset(com_lb, com_ub, host_lb, host_ub):
        logger.info("resetting growth rate constraint")
        # try out of order if it doesn't work
        try:
            com.variables.microbe_community_objective.lb = com_lb
            com.variables.microbe_community_objective.ub = com_ub
        except ValueError:
            com.variables.microbe_community_objective.ub = com_ub
            com.variables.microbe_community_objective.lb = com_lb
        
        try:
            com.variables.host_objective.lb = host_lb
            com.variables.host_objective.ub = host_ub
        except ValueError:
            com.variables.host_objective.ub = host_ub
            com.variables.host_objective.lb = host_lb

        logger.info(f'{com.variables.microbe_community_objective.name} lb and ub reset to {com.variables.microbe_community_objective.lb} and {com.variables.microbe_community_objective.ub}')
        logger.info(f'{com.variables.host_objective.lb} lb and ub reset to {com.variables.host_objective.lb} and {com.variables.host_objective.ub}')

    if context:
        # set the context reseter
        context(
            partial(
                reset,
                com.variables.microbe_community_objective.lb,
                com.variables.microbe_community_objective.ub,
                com.variables.host_objective.lb,
                com.variables.host_objective.ub,
            )
        )

    # compute the constraints from the growth dicts
    # it basically ammounts the the weighted sum of abundances and gorwth values
    # NOTE: Not perfect, bu we cannot get the primal since the model resets after the solution
    # TODO: maybe incorporate them in the solution somehow by fetching from primals before reset.

    logger.info(f'clamping host and community growth rates (margin = {rel_margin}) from solution: {growth}')
    com_obj_val: float = 0.0
    host_obj_val: float = 0.0
    for tax, val in dict(growth).items():
        if not np.isnan(val):
            if tax in com.microbe_taxa:
                com_obj_val += val * com.abundances[tax]
            else:
                host_obj_val += val * com.abundances[tax]
    logger.info(f'computed community growth = {com_obj_val} and host growth = {host_obj_val}')

    # print(com_obj_val, host_obj_val)  # debug

    # set lower and upper bounds to be within the relative margin
    for obj, val in [
        (com.variables.microbe_community_objective, com_obj_val),
        (com.variables.host_objective, host_obj_val),
    ]:
        # print(val)

        # set the lower bound
        new_lb = (1.0 - rtol) * val * (1.0 - rel_margin) - atol
        logging.info(f'setting lb of {obj.name} to {new_lb}')
        obj.lb = new_lb
        if obj.lb < atol:
            logger.info(
                "minimal growth rate smaller than tolerance," " setting to zero."
            )
            obj.lb = 0

        # set the upper bound
        new_ub = (1.0 + rtol) * val * (1.0 + rel_margin) + atol
        logging.info(f'setting ub of {obj.name} to {new_ub}')
        obj.ub = new_ub
        if obj.ub < atol:
            logger.info(
                "minimal growth rate smaller than tolerance," " setting to zero."
            )
            obj.ub = 0


def _clamp_growth(
    community: HostComModel,
    growth: Dict[str, float],
    rel_margin: float = 0.1,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    """Clamp the growth within +- rel_margin of the given growth rates"""
    context = get_context(community)

    def reset(taxon, lb, ub):
        logger.info("resetting growth rate constraint for %s" % taxon)
        community.constraints["objective_" + taxon].lb = lb
        community.constraints["objective_" + taxon].ub = ub

    for sp in community.taxa:
        logger.info("setting growth rate constraint for %s" % sp)
        obj = community.constraints["objective_" + sp]

        # set lower bound
        if context:
            context(partial(reset, sp, obj.lb, obj.ub))
        obj.lb = (1.0 - rtol) * growth[sp] * (1.0 - rel_margin) - atol
        if obj.lb < atol:
            logger.info(
                "minimal growth rate smaller than tolerance," " setting to zero."
            )
            obj.lb = 0

        # set the upper bound
        if context:
            context(partial(reset, sp, obj.lb, obj.ub))
        obj.ub = (1.0 + rtol) * growth[sp] * (1.0 + rel_margin) + atol
        if obj.ub < atol:
            logger.info(
                "minimal growth rate smaller than tolerance," " setting to zero."
            )
            obj.ub = 0
