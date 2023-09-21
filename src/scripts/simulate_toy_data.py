import sys
from pathlib import Path
import copy
from collections import OrderedDict

from cobra import Metabolite, Reaction, Model
import json


# append the src pathe to the syspath
src_path = str((Path(__file__).parent / '..' / '..' / 'src').resolve())
if src_path not in sys.path:
    sys.path.append(src_path)


# all imports
import simulator
from simulator.builder import format_results
from simulator.utils import set_all_seeds, get_current_seed


# ========= PARAMS =========
N_LEVELS = 2            # number of levels per condition
MARGINS = [0]           # sampling margin delta
MARGIN = 0.0            # sampling margin delta


N_SAMPLES = 100         # generated samples
RANGE_N_SAMPLES = 100   # samples for exp_designer
N_BATCHES = 1           # sub batches - probably do not touch
THINNING = 1000         # thinning param
CLAMP_ALL = False       # whether to clamp all biomasses

# min growths limits - per organism
# prevents numeric underflow
DESIGNER_MIN_GROWTH = 1e-6  
FINAL_MIN_GROWTH = 1e-5

N_PROC = 4              # number of threads to use - scale done if running out of memory

# path to save the data to -- 
OUTPUT_PATH = str((Path(__file__).parent / '..' / '..' / 'data' / 'simple_data').resolve())

SEED = 42               # random see
# ==========================


def create_model():
    """Create a single toy microorganism"""
    E_c = Metabolite('E_c', name='energy', compartment='c')
    M1_c = Metabolite('M1_c', name='metabolite1', compartment='c')
    M2_c = Metabolite('M2_c', name='metabolite2', compartment='c')

    e1_rxn = Reaction('M1_energy_rxn')
    e1_rxn.name = 'M1 energy creation'
    e1_rxn.lower_bound = 0.  # This is the default
    e1_rxn.upper_bound = 1000.  # This is the default
    e1_rxn.add_metabolites({
        M1_c: -1.0,
        E_c: 2
    })

    e2_rxn = Reaction('M2_energy_rxn')
    e2_rxn.name = 'M2 energy creation'
    e2_rxn.lower_bound = 0.  # This is the default
    e2_rxn.upper_bound = 1000.  # This is the default
    e2_rxn.add_metabolites({
        M2_c: -1.0,
        E_c: 1
    })

    e3_rxn = Reaction('M1_M2_energy_rxn')
    e3_rxn.name = 'M1 M2 energy creation'
    e3_rxn.lower_bound = 0.  # This is the default
    e3_rxn.upper_bound = 1000.  # This is the default
    e3_rxn.add_metabolites({
        M1_c: -1.0,
        M2_c: -2.0,
        E_c: 6
    })

    # add genes
    e1_rxn.gene_reaction_rule = '( G1 or  G11  )'
    e2_rxn.gene_reaction_rule = '( G2 or G11 )'
    e3_rxn.gene_reaction_rule = '( G3  )'

    # add import export
    M1_e = M1_c.copy()
    M1_e.name = 'M1_e'
    M1_e.id = 'M1_e'
    M1_e.compartment = 'e'

    M2_e = M2_c.copy()
    M2_e.name = 'M2_e'
    M2_e.id = 'M2_e'
    M2_e.compartment = 'e'

    t_m1_rxn = Reaction('t_M1_c')
    t_m1_rxn.name = 't_M1_c'
    t_m1_rxn.lower_bound = -1000.  # This is the default
    t_m1_rxn.upper_bound = 1000.  # This is the default
    t_m1_rxn.add_metabolites({
        M1_c: -1.0,
        M1_e: 1.0,
    })

    t_m2_rxn = Reaction('t_M2_c')
    t_m2_rxn.name = 't_M2_c'
    t_m2_rxn.lower_bound = -1000.  # This is the default
    t_m2_rxn.upper_bound = 1000.  # This is the default
    t_m2_rxn.add_metabolites({
        M2_c: -1.0,
        M2_e: 1.0,
    })

    # biomass
    bm_rxn = Reaction('Biomass_toy')
    bm_rxn.name = 'Biomass_toy'
    bm_rxn.lower_bound = -1000.  # This is the default
    bm_rxn.upper_bound = 1000.  # This is the default
    bm_rxn.add_metabolites({
        E_c: -1.0
    })


    model = Model('example_model')
    model.add_reactions([e1_rxn, e2_rxn, e3_rxn, t_m1_rxn, t_m2_rxn, bm_rxn])
    # add exchanges
    model.add_boundary(M1_e, type="exchange")
    model.add_boundary(M2_e, type="exchange")


    model.objective = 'Biomass_toy'
    return model


def insert_spec(com, exp_designer, specs):
    """Format the specifications into a json-outputtable format
    """
    results_full = OrderedDict()
    
    # log all inserted reactions so we don't reinsert them
    all_reactions = []
    
    # iterate them
    for name, exp_spec, pathway, coeff in specs:
        # get the results 
        res = simulator.add_pathway(com, exp_designer, name, exp_spec=exp_spec,
                                    pathway=pathway, prod_coeff=coeff, prev_rxns=all_reactions,
                                    microbe_coeff=coeff)

        formatted_res = format_results(res) 
        all_reactions += copy.copy(formatted_res['reactions'])
        # print(all_reactions)

        # add the metadata
        formatted_res['name'] = name
        formatted_res['spec'] = exp_spec

        # results
        results_full[name] = formatted_res
    
    return results_full  


# creat the community model
def create_community_model():
    """Assemble the community model"""

    # create identical model and give them names
    mods = [create_model(), create_model(), create_model(), create_model()]
    mods[0].id = 'H'
    mods[1].id = 'M0'
    mods[2].id = 'M1'
    mods[3].id = 'M2'

    # 0.98 + others - abundacnes
    dist = [0.98, 0.033, 0.033, 0.033]
    models = {
        m.id: m for m in mods
    }
    abundances = {
        m.id: v for m, v in zip(mods, dist)
    }
    
    # all but the first one are microbres
    microbes = list(m.id for m in mods)[1:]  

    # assemble models into the community model
    com = simulator.HostComModel(
        models=models,
        abundances=abundances, 
        microbe_taxa=microbes, 
        max_exchange=100
    )

    return com


def simulate_and_save(path_prefix, com, exp_designer, thinning, levels=3, conditions=2, results_full=None, margin=0.25):
    """Given a path prefix, simulate data and save everything"""

    # get the samples
    data = simulator.sample_across_conditions(com, exp_designer, conditions=conditions, levels=levels, 
                                              n_samples=N_SAMPLES, n_proc=N_PROC, 
                                              sim_kwargs=dict(min_growth=FINAL_MIN_GROWTH),
                                              samp_kwargs=dict(thinning=thinning, n_batches=N_BATCHES, 
                                                               clamp_all=CLAMP_ALL, margin=margin))


    metab_prod_fluxes = simulator.get_unsummed_metab_fluxes(com, data)  # metabolite production fluxes in medium
    biomass_fluxes = simulator.get_biomass_fluxes(com, data) 
    biomass_fluxes_rel = simulator.get_biomass_fluxes(com, data, relative=True) 
    internal_fluxes = simulator.get_internal_fluxes(com, data, 'H')  # for host
    internal_gene_fluxes = simulator.get_internal_gene_fluxes(com, data, 'H')  # for host

    # create destination
    Path(path_prefix).mkdir(parents=True, exist_ok=True)
    
    # pathway specs
    with open(path_prefix + 'pathways.json', 'w') as f:
        json.dump(results_full, f, indent=2, sort_keys=True)

    # csv data
    data.to_csv(path_prefix + 'raw_data.csv', index=False)  # raw fluxes
    metab_prod_fluxes.to_csv(path_prefix + 'metabs.csv', index=False)  # metabolites productions
    biomass_fluxes.to_csv(path_prefix + 'biomass.csv', index=False)  #  biomas fluxes
    biomass_fluxes_rel.to_csv(path_prefix + 'biomass_rel.csv', index=False)  # biomass fluxes normalized  0-1 per sample
    internal_fluxes.to_csv(path_prefix + 'internal.csv', index=False)  # internal fluxes of the host
    internal_gene_fluxes.to_csv(path_prefix + 'internal_gene.csv', index=False)  # host gene fluxes
    
    return data


def main():
    set_all_seeds(SEED)  # set seed for reproducibility

    # create a community model
    com = create_community_model()

    # add the pathway
    specs = [('P', ['D1 P', 'H P'], ['H', 'M1'], 1)]  # pathway "P"  - differential in 1
    result_full = insert_spec(com, exp_designer, specs)  # create the spec to save as json

    # create the xperimental designer
    exp_designer = simulator.ExperimentalDesigner(
        models=com, n_proc=N_PROC, n_samples=RANGE_N_SAMPLES, 
        range_n_samples=RANGE_N_SAMPLES, n_batches=N_BATCHES, 
        thinning=THINNING, save_samples=True,
        condition_limit=2,  # just 2  conditions
        min_growth=10**(-6),
        pre_scale=True,
        margin=MARGIN
    )

    # 
    simulate_and_save(OUTPUT_PATH, com = com, exp_designer=exp_designer, thinning=THINNING, levels=N_LEVELS, 
                      conditions=2, margin=MARGIN, results_full=result_full)






