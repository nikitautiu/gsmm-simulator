source('../R/evaluation.R')

N_PROC <- 10
SEED <- 42

future::plan('multisession', workers = N_PROC)  # parallelize across files
# future::plan('sequential')

# combined sets
files <- Sys.glob('../data/interim/final2/SET*') %>% 
  str_subset('.*SET[0-9][0-9]+_.*_T1000_MG.+') %>% 
  str_subset('_MG0$', negate = T)

print(files)

# all files
run_pipeline_mutli_sets(
  files = files,
  # output directly - skips having to rejoin them - saves a lot of time
  output_pattern = '../data/interim/separate_results/{name}_RESULT.gz.parquet',
  
  .progress = T,
  keep_max = 10,
  keep_features = c("bm", "metab", "gene", "internal"),
  seed = SEED,
  random_spec = c(F, T),  # run on both random specs and real specs
  
  # bootstrapping
  pipeline_func = \(raw_data, clean_data) { 
    # print('START')
    result <- run_pipeline_bootstrap(
      raw_data, clean_data,  
      n_rep = 30, 
      # n_samps = c(100), # 30 x(2, 5, 10, 25)
      n_samps = c(3, 5, 10, 100), # 30 x(3, 5, 10, 25)
      
      # multiple noise levels
      pipeline_func = \(raw_data, clean_data) { 
        run_pipeline_multi_noise(
          raw_data, clean_data, noise_lvls = c(0.0, 0.1, 0.5, 1.0),
          pipeline_func = ~ all_pipelines(.x, .y, kinds = c('ASCA', 'PCA', 'MOFA','MOFA_GROUPED', 'PLS', 'dummy')) # all pipelines
          # pipeline_func = ~ all_pipelines(.x, .y, kinds = c('dummy')) # all pipelines
        )
      }
    )
    # print('FINISH')
    result
  }
)
