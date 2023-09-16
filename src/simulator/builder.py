import copy
import itertools
from collections import OrderedDict, namedtuple
from collections.abc import Sized
from functools import partial
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cobra
import cobra.sampling
import joblib
import numpy as np
import math
import pandas as pd
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
from scipy.stats import nbinom
from tqdm import tqdm

from .community_model import HostComModel
from .designer import ExperimentalDesigner

logger = logging.getLogger('logger')


LB_EXPORT = 0

def make_models(n_micro: int, knockouts: Optional[Tuple[int, int]] = None, min_growth:float = 1e-4,  \
                knockout_genes:Optional[List[str]] = None):
    if knockouts is None:
        knockouts = (1, 4)  # mutants have between 1 and 4 knockouts by default

    # load the ecoli core model
    base_model: cobra.Model = load_model("textbook")

    logger.info("creating base models")

    # result lists
    microbes: List[str] = []
    models: List[cobra.Model] = []
    all_knockouts:List[List[str]]  = []

    idx = 0
    while idx < n_micro:
        # copy the base model so we don't mess something up
        # eg. irreversibly set some limit or smth
        mod = copy.deepcopy(base_model)

        # get the number of knockouts
        n_knock = np.random.randint(*knockouts, size=1)
        if knockout_genes is not None:
            n_knock = len(knockout_genes[idx])
            
        total_reacts = len(mod.reactions)
        mod.knockouts = []

        # get the reactions to knowck out
        while len(mod.knockouts) < n_knock:
            if knockout_genes is None:
                rxn_id = np.random.randint(total_reacts, size=1)[0]
                name = mod.reactions[rxn_id].id
            else:
                # use the passed knockouts
                name = knockout_genes[idx][len(mod.knockouts)]   

            # no biomass, export or already knocked-out reaction
            ok = (
                not (
                    name.lower().startswith("biomass") or name.lower().startswith("ex_")
                )
                and name not in mod.knockouts
            )
            if ok:
                mod.reactions.get_by_id(name).knock_out()
                mod.knockouts.append(mod.reactions.get_by_id(name).id)

        # all microbe names follow the M<number> naming convention
        mod.id = f"M{idx}"
        logger.info(f"model {mod.id} knockout: {mod.knockouts}")
        # print(mod.knockouts)

        # too many knockouts may violate some constraints in the ecoli model
        # make sure this is not the case, by attempting optimization
        sol = mod.optimize()
        if sol.status == OPTIMAL and sol.objective_value > min_growth:
            microbes.append(mod.id)
            models.append(mod)
            idx += 1  # only append if model is feasible

            all_knockouts.append(copy.copy(mod.knockouts))
        else:
            logger.info(f"model with knockouts {mod.knockouts} is infeasible, retry")
            if knockout_genes is not None:
                print(f'Knockouts {knockout_genes[idx]} failed')
                raise ValueError(f'Knockouts {knockout_genes[idx]} failed')
            # print(f'model with knockouts {mod.knockouts} is infeasible, retry')

    # get the distributions
    # dist = nbinom.rvs(5, 0.5, size = n_micro)
    dist = np.ones((n_micro,))  # same percentage for all, total sum of microbes = 2%
    dist = dist / np.sum(dist) * 0.02
    dist = [0.98] + list(dist)

    # return the models with distribtions
    base_model.id = "H"
    return [base_model] + models, dist, microbes, all_knockouts


def add_metabolite(
    com: HostComModel,
    model_name: str,
    name,
    name_prefix="Fantasy Metabolite",
    ext=True,
    imp=True,
    synth=True,
    skip_exist=True,
    max_transport: Optional[float] = None,
):
    """Add a fantasy metabolite to the model.
    If ext is true it means that there will be a membrane exchange for it with an unconstrained
    import export reaction. If ext is False, only export will be possible. If synth is True
    it means that the metabolite is synthetic and should not be created into the extracellular
    medium (aka the outer exchange reaction will only allow export)."""

    if max_transport is None:
        max_transport = com.max_exchange

    # check for existence
    if name + f"_c__{model_name}" in [str(m.id) for m in com.metabolites]:
        if not skip_exist:
            raise ValueError(f"{name} already part of the model")
        else:
            return  # skip everything

    full_name = name_prefix + name

    # add cytosolic and extracellular metabolite
    metab_c = cobra.core.Metabolite(
        f"{name}_c__{model_name}",
        formula=None,
        name=full_name,
        compartment=f"c__{model_name}",
    )
    metab_c.global_id = metab_c.id
    metab_c.community_id = model_name

    # and its extracellular counterpart
    metab_e = metab_c.copy()
    metab_e.id = f"{name}_e__{model_name}"
    metab_e.global_id = metab_e.id
    metab_e.compartment = f"e__{model_name}"
    metab_e.community_id = model_name

    # and its medium counterpart
    metab_m = metab_c.copy()
    metab_m.id = f"{name}_m"
    metab_m.global_id = metab_e.id
    metab_m.compartment = "m"
    metab_m.community_id = "m"

    # add to the model
    com.add_metabolites([metab_c, metab_e, metab_m])

    # add transport reactions
    lb, ub = max_transport, max_transport
    if not ext:
        ub = 0  # import is negative so 0 means to import
    if not imp:
        lb = 0

    # add the cytosol import reaction
    t_rxn = cobra.core.Reaction(
        id=f"IMP_{name}_c__{model_name}",
        name=name + " cytosol import",
        lower_bound=0,
        upper_bound=lb,
    )
    t_rxn.community_id = model_name
    t_rxn.global_id = t_rxn.id
    com.add_reactions([t_rxn])
    t_rxn.add_metabolites({metab_c.id: 1, metab_e.id: -1})

    # and the cytsol export one
    t_rxn = cobra.core.Reaction(
        id=f"EXP_{name}_c__{model_name}",
        name=name + " cytosol export",
        lower_bound=min(LB_EXPORT, ub),
        upper_bound=ub,
    )
    t_rxn.community_id = model_name
    t_rxn.global_id = t_rxn.id
    com.add_reactions([t_rxn])
    t_rxn.add_metabolites({metab_c.id: -1, metab_e.id: 1})

    # and the e <-> m exchanges
    # add the cytosol import reaction
    t_rxn = cobra.core.Reaction(
        id=f"IMP_{name}_e__{model_name}",
        name=name + " extracellular import",
        lower_bound=0,
        upper_bound=lb,
    )
    t_rxn.community_id = model_name
    t_rxn.global_id = t_rxn.id
    com.add_reactions([t_rxn])
    # same logic but adjusted for abundaces
    t_rxn.add_metabolites({metab_e.id: 1, metab_m.id: -com.abundances[model_name]})

    # and the cytsol export one
    t_rxn = cobra.core.Reaction(
        id=f"EXP_{name}_e__{model_name}",
        name=name + " extracellular export",
        lower_bound=min(LB_EXPORT, com.max_exchange) if ext else 0,
        upper_bound=com.max_exchange if ext else 0,
    )
    t_rxn.community_id = model_name
    t_rxn.global_id = t_rxn.id
    com.add_reactions([t_rxn])
    # same logic but adjusted for abundaces
    t_rxn.add_metabolites({metab_e.id: -1, metab_m.id: com.abundances[model_name]})

    # external exchange now
    try:
        ex_rxn = com.add_boundary(
            com.metabolites.get_by_id(metab_m.id), type="exchange"
        )
        ex_rxn.global_id = ex_rxn.id
        ex_rxn.community_id = model_name

        # stop import if synthetic
        if synth:
            ex_rxn.bounds = (0, 1000)
    except ValueError:
        pass  # skip if it exists

    return metab_m


def add_intracellular_sink(
    com: HostComModel, model_name: str, name: str
) -> cobra.Reaction:
    """Given a global model add an intracellular sink for the specified metabolite. Return the reaction"""

    rxn_name = f"SINK_{name}_c__{model_name}"
    metab = com.metabolites.get_by_id(f"{name}_c__{model_name}")
    rxn = com.add_boundary(metab, type="sink", lb=0, reaction_id=rxn_name)
    rxn.global_id = rxn.id
    rxn.community_id = model_name
    return rxn


def combine_reaction(
    com: HostComModel,
    model_name: str,
    existing_rxn: str,
    new_rxn: Mapping[str, float],
    coeff=1,
    compartment: str = "c",
) -> cobra.Reaction:
    """Given a reaction in the form of a reaction stoichimoetry dict or existing reaction.

    Attach it to an exsiting reaction while multiplying the stoichiometric coefficients
    with the coefficient. By using a negative number, we can add it ithe oposite direction.
    Useful if we are targeting  a negative flux reaction.

    Beware, expects the metabolites to exist in the model already.
    If the suffix is not present on the metabolites add it. Makes it easier to
    when adding in bulk. Reactions are directly inserted into cytosol compart.
    """

    stoic_dict = new_rxn
    if not isinstance(new_rxn, dict):
        # get the stoichiometry
        stoic_dict = new_rxn.get_coefficients(new_rxn.metabolites)

    # add suffixes if necessary
    stoic_dict = {
        f"{metab}_{compartment}__{model_name}": c for metab, c in stoic_dict.items()
    }

    # multiply coefficients
    stoic_dict = {metab: c * coeff for metab, c in stoic_dict.items()}

    # get the existing reaction
    com.reactions.get_by_id(f"{existing_rxn}__{model_name}").add_metabolites(stoic_dict)
    return com.reactions.get_by_id(f"{existing_rxn}__{model_name}")


def add_cost_to_export(
    com: HostComModel, model_name: str, name: str, cost_name: str, coeff: float = 1
) -> cobra.Reaction:
    """Given a model name the name of a synthetic metabolite, and the name of a metabolite to associate with its export
    add a cost to the export  of the metabolite"""

    # add the cost to the cytosolic export reaction
    return combine_reaction(
        com, model_name, f"EXP_{name}_c", new_rxn={cost_name: -coeff}, compartment="c"
    )


def add_cost_to_sink(
    com: HostComModel, model_name: str, name: str, cost_name: str, coeff: float = 1
) -> cobra.Reaction:
    """Given a model name the name of a synthetic metabolite, and the name of a metabolite to associate with its export
    add a cost to the export  of the metabolite. If the coeff is negative, sink produces the metab, otherwise, it consumes it
    """

    # add the cost to the cytosolic export reaction
    return combine_reaction(
        com, model_name, f"SINK_{name}_c", new_rxn={cost_name: -coeff}, compartment="c"
    )


def force_export(
    com: HostComModel, model_name: str, metab_name: str, ex_rate: float = 0.9
):
    """Given a model and a metabolite name, add an artifical constraint forcing export.

    The export flux will have to be at least ex_rate times the total production flux.
    This is for the export from the cytosol to the extracellular space
    """

    constr_name = f"exp_force_{metab_name}__{model_name}"
    full_metab_name = f"{metab_name}_c__{model_name}"
    exp_rxn_name = f"EXP_{metab_name}_c__{model_name}"
    imp_rxn_name = f"IMP_{metab_name}_c__{model_name}"

    if constr_name in com.constraints:
        # remove it first
        com.remove_cons_vars([com.constraints[constr_name]])

    prod_flux = Zero  # symbolic math exp
    for rxn in list(com.metabolites.get_by_id(full_metab_name).reactions):
        # exclude the transport reaction from the computations
        if not rxn.id in [exp_rxn_name]:
            coeff = rxn.get_coefficient(
                full_metab_name
            )  # coeffs to determine prod direction

            # add to production flux expression
            prod_flux += coeff * rxn.forward_variable - coeff * rxn.reverse_variable

    # get the transport reaction - we use the _t convention for the names
    t_rxn = com.reactions.get_by_id(exp_rxn_name)

    # get the direction of the reaction
    coeff = t_rxn.get_coefficient(full_metab_name)  # coeffs to determine prod direction
    t_exp = -coeff * t_rxn.forward_variable + coeff * t_rxn.reverse_variable

    constr_exp = ex_rate * prod_flux - t_exp

    # now we com to make sure that export flux is at least ex_rate times of the production flux
    constr = model.problem.Constraint(constr_exp, name=constr_name, ub=0.0, lb=-1e9)

    # add it to the model
    com.add_cons_vars([constr])


def force_import(
    com: HostComModel, model_name: str, metab_name: str, imp_rate: float = 0.9
):
    """Given a model and a metabolite name, add an artifical constraint forcing import.

    The import flux will have to be at least ex_rate times the total production flux.
    This is for the export from extracellular space to the cytosol
    """

    constr_name = f"imp_force_{metab_name}__{model_name}"
    full_metab_name = f"{metab_name}_c__{model_name}"
    full_metab_name_e = f"{metab_name}_e__{model_name}"
    full_med_metab_name = f"{metab_name}_m"
    imp_rxn_name = f"IMP_{metab_name}_c__{model_name}"
    imp_rxn_name_e = f"IMP_{metab_name}_e__{model_name}"
    med_ex_name = f"EX_{metab_name}_m"  # medium exchange reaction
    exp_rxn_prefix = f"EXP_{metab_name}_c"  # prefix of production reactions

    if constr_name in com.constraints:
        # remove it first
        com.remove_cons_vars([com.constraints[constr_name]])

    prod_flux = Zero  # symbolic math exp
    for rxn in list(com.metabolites.get_by_id(full_med_metab_name).reactions):
        # just export reactions
        if not rxn.id.startswith("IMP_") and not rxn.id.startswith("EX_"):
            coeff = rxn.get_coefficient(
                full_med_metab_name
            )  # coeffs to determine prod direction

            # add to production flux expression
            prod_flux += coeff * rxn.forward_variable - coeff * rxn.reverse_variable

    # ADD THE ONE FOR THE INTERNAL FLUX
    # get the transport reaction
    t_rxn = com.reactions.get_by_id(imp_rxn_name)
    # minus to make the computations work
    abund_coeff = -com.reactions.get_by_id(imp_rxn_name_e).get_coefficient(
        full_med_metab_name
    )

    # get the direction of the reaction
    imp_coeff = com.reactions.get_by_id(imp_rxn_name).get_coefficient(
        full_metab_name
    )  # get the metabolite name
    t_imp = imp_coeff * t_rxn.forward_variable - imp_coeff * t_rxn.reverse_variable

    constr_exp = imp_rate * prod_flux - abund_coeff * t_imp

    # now we com to make sure that import flux is at least imp_rate times of the production flux
    constr = com.problem.Constraint(constr_exp, name=constr_name, ub=0, lb=-1e16)

    # add it to the model
    com.add_cons_vars([constr])


import re


# def get_targets(exp_designer, spec, model):
#     targets = exp_designer.get_target(spec, model=model).sort_values(
#         "rank_prod", ascending=False
#     )
#     if 'flag' in targets.columns:
#         targets = targets.loc[targets.flag == 1]

#     return list(
#         targets[
#             ~targets.reaction.str.lower().str.startswith("biomass")
#             & ~targets.reaction.str.lower().str.startswith("ex")
#         ].reaction
#     )

from collections import namedtuple


def get_targets(exp_designer, spec, model):
    targets = exp_designer.get_target(spec, model=model).sort_values(
        "rank_prod", ascending=False
    )
    # subset just hte ones with the flag
    # if 'flag' in targets.columns:
        # targets = targets.loc[targets.flag == 1]

    return targets.loc[
        ~targets.reaction.str.lower().str.startswith("biomass")
        & ~targets.reaction.str.lower().str.startswith("ex")
    ]


def get_cost_targets(exp_designer, spec, model):
    targets = exp_designer.get_target(spec, model=model, by="metabolite").sort_values(
        "rank_prod", ascending=False
    )
    targets.metabolite = targets.metabolite.str.replace(
        r"_[^_]+$", r"", regex=True
    )  # ugly workaround
    return targets


def sample_targets(targets: pd.DataFrame, n: int):
    """Sample from the top n reactions. Use the normalized ranks as probabilites"""
    pot_targets = targets.iloc[:n].copy()  # top n

    pot_targets["prob"] = (
        pot_targets["rank_prod"] / pot_targets["rank_prod"].sum()
    )  # convert to probs
    vals = (
        pot_targets.reaction
        if "reaction" in pot_targets.columns
        else pot_targets.metabolite
    )

    # use the probabilites
    # returns a list of 1
    return np.random.choice(vals, size=1, p=pot_targets.prob)[0]


def add_pathway(
    com: HostComModel,
    exp_designer: ExperimentalDesigner,
    name: str,
    exp_spec: Union[str, List[str]],
    pathway: List[str],
    samp_from: int = 5,
    prod_coeff: float = 1.0,
    microbe_coeff: float = 1e-7,
    prev_rxns: Optional[List[str]] = None,
    use_biomass_for_microbes: bool = True
) -> Dict[str, Any]:
    """Insert a pathway into the community model that has a given differential exp
    specification. The top reations to attach this to are sampled from the given top `samp_from` reactions

    Pathway is the list of orgnisms the pathway spans.
    The names are prefixed with name and start from 1.

    `prod_coeff` tells for every 1 substrate how much fantasy product is made. If the pathway has a `prod_coeff` == 2
    and 3 steps, at each step the product is multiplied by 2, ending up 8 times higher than the amount it starts.
    """

    # actual reactions and metabolites inserted
    # also sinks
    reactions = []
    metabs = []
    sinks = []
    ext_metabs = []
    coeffs = []
    cost_metabs = []
    organisms = []

    base_coeff = prod_coeff
    base_microbe_coeff = microbe_coeff

    # intialize previous reactions
    if prev_rxns is None:
        prev_rxns = []

    # make it a list of repeating specs if not pased as list
    if not isinstance(exp_spec, list):
        exp_spec = [exp_spec] * len(pathway)

    last_metab = None
    for idx, (organism, exp_spec) in enumerate(zip(pathway, exp_spec)):
        organisms.append(organism)

        # coeff for simple or mixed pathway
        if not all(x == 'H' for x in pathway) and organism == 'H':
            prod_coeff = base_microbe_coeff
        else:
            prod_coeff = base_coeff

        # check for organism change
        first = idx == 0
        last = idx == len(pathway) - 1

        # whether to import or export the metab
        is_import = not first and pathway[idx] != pathway[idx - 1]
        is_export = last or pathway[idx] != pathway[idx + 1]

        metab_name: str = f"{name}{idx+1}"  # get the metabolite name

        cost_targets = get_cost_targets(
            exp_designer, "H", organism
        )  # high cost in organism

        # reaction targets and cost targets
        unfiltered_rxn_targets = get_targets(exp_designer, exp_spec, organism)

        coef_sign = 1
        # keep only those with the correct flag
        rxn_targets = unfiltered_rxn_targets.loc[unfiltered_rxn_targets.flag]
        if len(rxn_targets) == 0:
            new_spec = exp_spec.replace('P', 'N')  # try with negative
            print(f'Trying with new spec: {exp_spec} -> {new_spec}')
            coef_sign = -1

            unfiltered_rxn_targets = get_targets(exp_designer, exp_spec, organism)
            rxn_targets = unfiltered_rxn_targets.loc[unfiltered_rxn_targets.flag]

        if len(rxn_targets) == 0:
            print('Failed with both specs')
            raise ValueError('Failed with both specs')

        # sample for the best target without existing reactions
        existing_rxns = [rxn.id for rxn in reactions]

        # print(existing_rxns)
        if ((organism != 'H' and 'H' in pathway)  or ('H' not in pathway)) and use_biomass_for_microbes:
            rxn_targ = 'Biomass_Ecoli_core'  # for microbes use the ecoli core biomass
        else:
            rxn_targ = sample_targets(
                rxn_targets[~rxn_targets.reaction.isin(existing_rxns)], 
                samp_from
            )
        cost_targ = sample_targets(cost_targets, samp_from)

        # print(rxn_targ, cost_targ)

        # add the previous metab
        if is_import:
            metab = add_metabolite(
                com,
                organism,
                last_metab,
                name_prefix="",
                synth=True,
                ext=False,
                imp=True,
            )

        # add the metabolite
        metab = add_metabolite(
            com,
            organism,
            metab_name,
            name_prefix="",
            synth=True,
            ext=is_export,
            imp=False,
        )
        metabs.append(metab)

        # sink
        if is_export:
            sink = add_intracellular_sink(com, organism, metab_name)
            sinks.append(sink)
            ext_metabs.append(metab)
        # reaction - conditional based on wether it's the first one or not
        if last_metab is None:
            rxn = combine_reaction(com, organism, rxn_targ, {metab_name: coef_sign * prod_coeff})
        else:
            # 
            rxn = combine_reaction(
                com, organism, rxn_targ, {
                    # ensure different sign coefficients
                    last_metab: -np.sign(coef_sign * prod_coeff), 
                    metab_name: coef_sign * prod_coeff
                }
            )
        
        coeffs.append(coef_sign * prod_coeff)
        reactions.append(rxn)  # add to the reaction list
        last_metab = metab_name  #  so it's used in the next iteration

        # force export if necessary
        if is_export:
            add_cost_to_sink(com, organism, metab_name, cost_targ)
            cost_metabs.append(cost_targ)  # or targ
        else:
            cost_metabs.append(None)  # none

    return {
        "reactions": reactions,
        "metabolites": metabs,
        "sinks": sinks,
        "external_metabs": ext_metabs,
        "prod_coeffs": coeffs,
        "cost_metabs": cost_metabs,
        "organisms": organisms
    }






def add_fixed_pathway(
    com: HostComModel,
    name: str,
    pathway: List[Tuple[Union[str, float]]],  
) -> Dict[str, Any]:
    """Insert a pathway into the community model that has a given differential exp
    specification. The top reations to attach this to are sampled from the given top `samp_from` reactions

    Pathway is the list of orgnisms the pathway spans.
    The names are prefixed with name and start from 1.

    `prod_coeff` tells for every 1 substrate how much fantasy product is made. If the pathway has a `prod_coeff` == 2
    and 3 steps, at each step the product is multiplied by 2, ending up 8 times higher than the amount it starts.
    """

    # actual reactions and metabolites inserted
    # also sinks
    reactions = []
    metabs = []
    sinks = []
    ext_metabs = []
    coeffs = []
    cost_metabs = []
    organisms = []

    last_metab = None
    for idx, param_tup in enumerate(pathway):
        # add or not a parameterr that lets you specify the in coefficient for hte reaction
        in_coeff = 1
        if len(param_tup) == 4:
            (organism, reaction, cost_metab, coeff) = param_tup
        else:
            (organism, reaction, cost_metab, in_coeff, coeff) = param_tup

        organisms.append(organism)

        # check for organism change
        first = idx == 0
        last = idx == len(pathway) - 1

        # whether to import or export the metab
        is_import = not first and pathway[idx][0] != pathway[idx - 1][0]  # check the organism in this case
        is_export = last or pathway[idx][0] != pathway[idx + 1][0]

        metab_name: str = f"{name}{idx+1}"  # get the metabolite name


        # get targets by name
        rxn_targ = reaction
        cost_targ = cost_metab

        print(rxn_targ, cost_targ)

        # add the previous metab
        if is_import:
            metab = add_metabolite(
                com,
                organism,
                last_metab,
                name_prefix="",
                synth=True,
                ext=False,
                imp=True,
            )

        # add the metabolite
        metab = add_metabolite(
            com,
            organism,
            metab_name,
            name_prefix="",
            synth=True,
            ext=is_export,
            imp=False,
        )
        metabs.append(metab)

        # sink
        if is_export:
            sink = add_intracellular_sink(com, organism, metab_name)
            sinks.append(sink)
            ext_metabs.append(metab)
        # reaction - conditional based on wether it's the first one or not
        if last_metab is None:
            rxn = combine_reaction(com, organism, rxn_targ, {metab_name: coeff})
        else:
            rxn = combine_reaction(
                com, organism, rxn_targ, {
                    # different signs
                    last_metab: -np.sign(coeff) * in_coeff, 
                    metab_name: coeff
                }
            )
        reactions.append(rxn)  # add to the reaction list
        last_metab = metab_name  #  so it's used in the next iteration
        coeffs.append(coeff)

        # force export if necessary
        if is_export:
            add_cost_to_sink(com, organism, metab_name, cost_targ)
            cost_metabs.append(cost_targ)  # or targ
        else:
            cost_metabs.append(None)  # none

    return {
        "reactions": reactions,
        "metabolites": metabs,
        "sinks": sinks,
        "external_metabs": ext_metabs,
        "prod_coeffs": coeffs,
        "cost_metabs": cost_metabs,
        "organisms": organisms
    }



def format_results(results: Dict[str, List[Any]]):
    """Given the results from a pathway convert them to a nested dictionary structure that can be persisted in json"""
    rxns_global = [rxn.global_id for rxn in results["reactions"]]
    rxns = [rxn.id for rxn in results["reactions"]]
    metabs = [metab.id for metab in results["metabolites"]]
    metab_names = [metab.name for metab in results["metabolites"]]
    metabs_ext = [metab.id for metab in results["external_metabs"]]
    metab_names_ext = [metab.name for metab in results["external_metabs"]]
    genes = [list(rxn.gpr.genes) for rxn in results["reactions"]]
    cost_metabs = [metab for metab in results["cost_metabs"]]
    prod_coeffs = results['prod_coeffs']
    organisms = results['organisms']

    host_genes = sum(
        [
            list(rxn.gpr.genes)
            for rxn in results["reactions"]
            if rxn.id.lower().endswith("_h")
        ],
        [],
    )

    return {
        "reactions": rxns,
        "reactions_global": rxns_global,
        "metabolites": metabs,
        "metabolite_names": metab_names,
        "metabolites_ext": metabs_ext,
        "metabolite_names_ext": metab_names_ext,
        "genes": genes,
        "host_genes": host_genes,
        "cost_metabs": cost_metabs,
        "prod_coeffs": prod_coeffs,
        "organisms": organisms
    }


def extract_hardcoded_pathways(results):
    """Given a string formatted full_results description as in a json saved for the setup,
    recreate the necessary specs to pass to the add_fixed_pathway function"""
    return [
        [
            (
                path_name, 
                list(zip(path['organisms'], path['reactions_global'], path['cost_metabs'], path['prod_coeffs']))
            ) 
            for path_name, path in res.items()
        ] 
        for res in results
    ]
