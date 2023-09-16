import copy
import itertools
from collections import OrderedDict
from collections.abc import Sized
from functools import partial
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd


from .community_model import HostComModel, _sample_across_conditions
from .designer import ExperimentalDesigner, limit_combinations

logger = logging.getLogger("logger")


def sample_across_conditions(
    com: HostComModel,
    designer: ExperimentalDesigner,
    conditions: Optional[Union[int, List[str]]] = None,
    levels: int = 2,
    n_samples: int = 1000,
    n_proc: int = 1,
    sim_kwargs: Dict[str, Any] = {},
    samp_kwargs: Dict[str, Any] = {},
):
    """
    Given a community model, an experimental designer for those models, return dataframes with samples across
    the existing experimental conditions.

    By default, uses all conditions. If `conditions` is an int, just uses the first `conditions`.
    Levels specifies how many levels each condition should use. By the default, the factors are binary.

    Additional params can be passed to the simulator and sampler functions of the community.
    """
    limit_specs_and_lims = list(
        limit_combinations(designer.exp_factors, conditions, levels=levels)
    )

    # proxy to the other fucntion
    return _sample_across_conditions(
        com=com,
        limit_specs_and_lims=limit_specs_and_lims,
        n_samples=n_samples,
        n_proc=n_proc,
        sim_kwargs=sim_kwargs,
        samp_kwargs=samp_kwargs,
    )


def get_biomass_fluxes(
    com: HostComModel,
    samp: pd.DataFrame,
    just_microbes: bool = False,
    relative: bool = False,
):
    """Given a sample, extract the biomass fluxes for all the organisms.

    If just_microbes, drop the host.
    If relative, do relative biomasses.
    """
    # get just the biomass reactions
    biomass_df = samp.loc[samp["reaction"].str.startswith("Biomass")].copy()

    # extract organism
    biomass_df.loc[:, "organism"] = biomass_df["reaction"].str.replace(
        r"^.*__(.+)$", r"\1", regex=True
    )
    # biomass_df.drop('reaction', axis='columns', inplace=True)
    biomass_df = biomass_df.reset_index(drop=True)  # make it cleaner

    # keep only microbes or not
    if just_microbes:
        biomass_df: pd.DataFrame = biomass_df.loc[
            biomass_df["organism"].isin(com.microbe_taxa)
        ]

    # do relative abundances
    if relative:
        group_cols = [
            col
            for col in biomass_df.columns
            if col not in ["reaction", "flux", "organism"]
        ]
        grouped_df = biomass_df.groupby("samp")
        biomass_df = grouped_df.apply(
            lambda x: x.assign(flux=x.flux / x.flux.sum())
        ).reset_index(drop=True)

    return biomass_df


def get_internal_fluxes(
    com: HostComModel, samp: pd.DataFrame, organism: str, compartment: str = "c"
):
    """Get the internal fluxes of the specified organism from the specified organism"""

    full_compart = f"{compartment}__{organism}"  # micom format for global comparts
    # get just reactions that span that compartment. this way we avoid exchanges, then clear sinks and demands as well
    # also trim the biomass one
    rxns = com.reactions.query(
        lambda rxn: {full_compart} == rxn.compartments
        and rxn not in com.boundary
        and not rxn.id.lower().startswith("biomass")
        and not rxn.id.lower().startswith("sink_")
    )
    rxn_names = [rxn.id for rxn in rxns]

    # subset the sample
    internal_fluxes = samp.loc[samp["reaction"].isin(rxn_names)].reset_index(drop=True)
    return internal_fluxes


def get_internal_gene_fluxes(
    com: HostComModel, samp: pd.DataFrame, organism: str, compartment: str = "c"
):
    """Get the internal fluxes of the specified organism from the specified organism"""

    full_compart = f"{compartment}__{organism}"  # micom format for global comparts
    # get just reactions that span that compartment. this way we avoid exchanges, then clear sinks and demands as well
    # also trim the biomass one
    rxns = com.reactions.query(
        lambda rxn: {full_compart} == rxn.compartments
        and rxn not in com.boundary
        and not rxn.id.lower().startswith("biomass")
        and not rxn.id.lower().startswith("sink_")
    )

    # get the genes associated with each one
    gene_to_rxn = [
        {"reaction": rxn.id, "gene": gene} for rxn in rxns for gene in rxn.gpr.genes
    ]
    merged_df = pd.merge(
        pd.DataFrame.from_records(gene_to_rxn), samp, on="reaction", how="inner"
    )
    merged_df["abs_flux"] = merged_df["flux"].abs()
    merged_df.drop("reaction", axis="columns", inplace=True)  # drop reaction

    # group and compute sums, group by everything but
    group_cols = [col for col in merged_df.columns if col not in ["flux", "abs_flux"]]
    results = (
        merged_df.groupby(group_cols)
        .agg({"flux": "sum", "abs_flux": "sum"})
        .reset_index()
    )

    return results


def get_metab_production_fluxes(
    com: HostComModel, samp: pd.DataFrame, compartment: str = "m"
):
    """Given a host community model a sample and a compartment, return the production fluxes of each metabolite in the compartment"""
    # get the metabolites in the compartment
    metabs = com.metabolites.query(lambda x: x.compartment == compartment)
    # and the coefficients of all the reactions that feed into them
    coeff_df = [
        {
            "reaction": rxn.id,
            "metab": metab.global_id,
            "metab_name": metab.name,
            "coeff": rxn.get_coefficient(metab),
        }
        for metab in metabs
        for rxn in metab.reactions
    ]
    coeff_df = pd.DataFrame.from_records(coeff_df)
    # merge with sample and only keep positive fluxes
    merged_df = pd.merge(coeff_df, samp, on="reaction", how="inner")
    merged_df["flux"] = merged_df["flux"] * merged_df["coeff"]

    merged_df.drop(["coeff"], axis="columns", inplace=True)
 
    # drop the negative flux ones
    merged_df = merged_df.loc[
        merged_df["flux"] >= 0
    ]  # >=0 is crucial, 0 flux will be dropped otherwise

    # group by everything but reaction and flux
    group_cols = [col for col in merged_df.columns if col not in ["reaction", "flux"]]
    results = merged_df.groupby(group_cols).agg({"flux": "sum"}).reset_index()

    return results



def get_metab_production_fluxes(
    com: HostComModel, samp: pd.DataFrame, compartment: str = "m"
):
    """Given a host community model a sample and a compartment, return the production fluxes of each metabolite in the compartment"""
    # get the metabolites in the compartment
    metabs = com.metabolites.query(lambda x: x.compartment == compartment)
    # and the coefficients of all the reactions that feed into them
    coeff_df = [
        {
            "reaction": rxn.id,
            "metab": metab.global_id,
            "metab_name": metab.name,
            "coeff": rxn.get_coefficient(metab),
        }
        for metab in metabs
        for rxn in metab.reactions
    ]
    coeff_df = pd.DataFrame.from_records(coeff_df)
    # merge with sample and only keep positive fluxes
    merged_df = pd.merge(coeff_df, samp, on="reaction", how="inner")
    merged_df["flux"] = merged_df["flux"] * merged_df["coeff"]

    merged_df.drop(["coeff"], axis="columns", inplace=True)
 
    # drop the negative flux ones
    merged_df = merged_df.loc[
        merged_df["flux"] >= 0
    ]  # >=0 is crucial, 0 flux will be dropped otherwise

    # group by everything but reaction and flux
    group_cols = [col for col in merged_df.columns if col not in ["reaction", "flux"]]
    results = merged_df.groupby(group_cols).agg({"flux": "sum"}).reset_index()

    return results


def get_unsummed_metab_fluxes(
    com: HostComModel, samp: pd.DataFrame, compartment: str = "m"
):
    """Given a host community model a sample and a compartment, return the production fluxes of each metabolite in the compartment, unsummed, all reactions kept intact
    And organism information so we can weight it with the biomass"""
    # get the metabolites in the compartment
    metabs = com.metabolites.query(lambda x: x.compartment == compartment)
    # and the coefficients of all the reactions that feed into them
    coeff_df = [
        {
            "reaction": rxn.id,
            "metab": metab.global_id,
            "metab_name": metab.name,
            "coeff": rxn.get_coefficient(metab),
        }
        for metab in metabs
        for rxn in metab.reactions
    ]
    coeff_df = pd.DataFrame.from_records(coeff_df)
    # merge with sample and only keep positive fluxes
    merged_df = pd.merge(coeff_df, samp, on="reaction", how="inner")
    merged_df["flux"] = merged_df["flux"] * merged_df["coeff"]

    # merged_df.drop(["coeff"], axis="columns", inplace=True)
 
    # drop the negative flux ones
    merged_df.loc[
        merged_df["flux"] < 0, "flux"
    ] = 0 
    merged_df["organism"] = merged_df["reaction"].str.replace(r'^.*__([^_]+)$', '\\1', regex= True)

    return merged_df
