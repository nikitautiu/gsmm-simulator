library(tidyverse)
library(progressr)
library(arrow)
library(gASCA)

#' Process JSON specs
#'
#' @param data The specs as read from JSON
#'
#' @return A flat list of of all the features involved in this pathway
#' @export
#'
#' @examples
to_pathway_features <- function(data) {
  all_elems <- c(str_trim(list_c(as.list(data$genes))), 
                 data$reactions, data$organisms)
  
  if (!is.null(data$genes) && length(data$genes) > 0) {
    all_elems <- c(all_elems, data$genes[[1]])
  }
  
  all_elems <- c(all_elems, str_trim(data$metabolite_names_ext))
  
  # ugly wat to flatten a list into a vector 
  # agnostic to inputting lists or vectors
  as_vector(list_flatten(as.list(all_elems)))
}

#' Read the data from a directory, with specs and all
#'
#' @param base_path The path where the dataset is
#' @param keep_raw Wether to keep the raw data
#'
#' @return A list with the data as individual tibbles as well as metadata.
#' @export
#'
#' @examples
load_data <- function(base_path, keep_raw=F) {
  sim_specs <- jsonlite::fromJSON(glue::glue('{base_path}/pathways.json'))
  full_data <- read_csv(glue::glue('{base_path}/raw_data.csv'))
  raw_bm_data <- read_csv(glue::glue('{base_path}/biomass.csv'))
  raw_metab_data <- read_csv(glue::glue('{base_path}/metabs.csv'))
  internal_data <- read_csv(glue::glue('{base_path}/internal.csv'))
  internal_gene_data <- read_csv(glue::glue('{base_path}/internal_gene.csv'))
  
  # extract_metadata from path
  directory <- base_path %>% str_extract('/?([^/]+$)', 1)
  dataset_meta <- tribble(~ dataset_full_name, directory) %>% 
    mutate(
      dataset_set = str_extract(dataset_full_name, '^([^_]+)', 1), 
      dataset_levels = str_extract(dataset_full_name, '_L([0-9]+)', 1) %>%  as.numeric(),
      dataset_microbes = str_extract(dataset_full_name, '_M([0-9]+)', 1) %>%  as.numeric(),
      dataset_margin = str_extract(dataset_full_name, '_MG(.+)$', 1) %>%  as.numeric(),
    )
  
  # create spec object
  raw_specs <- sim_specs
  sim_specs <- map(sim_specs, to_pathway_features)
  
  # keep all but 'H' in bm_data
  bm_data <- raw_bm_data %>% 
    filter(organism != 'H') 
  
  # clean up metab data
  metab_data <-  raw_metab_data %>%
    mutate(flux = ifelse(flux > 0, flux, 0)) %>% 
    group_by(across(-c(reaction, coeff, flux, organism))) %>%
    summarise(flux = sum(flux)) %>%
    ungroup() 
  
  
  data <- list(
    "specs" = sim_specs,
    "raw_specs" = raw_specs,
    "bm_data" = bm_data,
    "metab_data" = metab_data,
    "internal_data" = internal_data,
    "gene_data" = internal_gene_data,
    "dataset_meta" = dataset_meta
  )
  
  if (keep_raw) {
    data[["full_data"]] = full_data
    data[["raw_bm_data"]] = raw_bm_data
    data[["raw_metab_data"]] = raw_metab_data
  }
  
  # return the dat
  data
}



clean_up_data <- function(raw_data) {
  # extract from the data
  metab_data <- raw_data$metab_data
  internal_data <- raw_data$internal_data
  internal_gene_data <- raw_data$gene_data
  bm_data <- raw_data$bm_data
  
  # get grouping cols
  cols <- raw_data$internal_data %>% colnames() 
  piv_cols <- cols[3:length(cols)]
  
  clean_metab_data <- metab_data %>% 
    pivot_wider(id_cols = piv_cols, names_from = metab_name, values_from = flux) %>% 
    replace(is.na(.), 0) %>% 
    mutate(across(-piv_cols, scale))
  
  clean_internal_data <- internal_data %>% 
    pivot_wider(id_cols = piv_cols, names_from = reaction, values_from = flux) %>% 
    replace(is.na(.), 0) %>% 
    mutate(across(-piv_cols, scale))
  
  clean_gene_data <- internal_gene_data %>% 
    pivot_wider(id_cols = piv_cols, names_from = gene, values_from = flux) %>% 
    replace(is.na(.), 0) %>% 
    mutate(across(-piv_cols, scale))
  
  clean_bm_data <- bm_data %>% 
    pivot_wider(id_cols = piv_cols, names_from = organism, values_from = flux) %>% 
    replace(is.na(.), 0) %>% 
    mutate(across(-piv_cols, scale))
  
  
  clean_full_data <- clean_metab_data %>% 
    inner_join(clean_internal_data, by=piv_cols) %>% 
    inner_join(clean_gene_data, by=piv_cols) %>% 
    inner_join(clean_bm_data, by=piv_cols) 
  
  clean_full_data <- clean_full_data %>% 
    replace(is.na(.), 0) 
  
  col_types <- c(
    # rep('meta', length(piv_cols)),
    rep('metab', dim(distinct(metab_data, metab))[1]),
    rep('internal', dim(distinct(internal_data, reaction))[1]),
    rep('gene', dim(distinct(internal_gene_data, gene))[1]),
    rep('bm', dim(distinct(bm_data, organism))[1])
  )
  
  list(
    data = clean_full_data,
    meta_cols = piv_cols,
    data_cols = setdiff(colnames(clean_full_data), piv_cols),
    col_types = col_types,
    specs = raw_data$specs
  )
}

# used for shuffling
permute_seq <- function(x) {
  n <- length(x)
  for (i in seq_along(x)) {
    J <- i + floor((0.69 * (i %% 3) + 0.01) / 2 * (n - i + 1))
    x[c(i, J)] <- x[c(J, i)]
  }
  return(x)
}


subset_features <- function(clean_data, feats, imp_feats = NULL, keep_max = 20) {
  # if imp_feats is passed, we keep the important features and deteriministically
  # add some others until we reach keep_max

  data <- rlang::duplicate(clean_data)

  if (!is.null(imp_feats))
    imp_feats <- imp_feats %>% list_c()

  kept_data_cols <- map(feats, ~ {
    # get the feature cols for that one
    feat_cols <- data$data_cols[data$col_types %in% .x]
    if (!is.null(imp_feats)) {
      imp_cols <- feat_cols[feat_cols %in% imp_feats]
      remaining_cols <- setdiff(feat_cols, imp_cols)

      # subsample cols deterministically
      n_to_keep <- keep_max - length(imp_cols)
      remaining_cols <- remaining_cols[permute_seq(1:length(remaining_cols)) %>%  head(n_to_keep)]

      feat_cols <- feat_cols[feat_cols %in% c(imp_cols, remaining_cols)]
    }
    # return the preserved featcols
    feat_cols
  }) %>% list_c()  # concat

  # get a mask with the meta cols and the kept ones
  feat_mask <- colnames(data$data) %in% c(
    data$meta_cols,
    data$data_cols[data$data_cols %in% kept_data_cols]
  )

  new_data_cols <- data$data_cols[data$data_cols %in% kept_data_cols]
  new_col_types <- data$col_types[data$data_cols %in% kept_data_cols]
  new_data <- data$data[, feat_mask]

  data$data <- new_data
  data$data_cols <- new_data_cols
  data$col_types <- new_col_types

  data
}


add_noise <-  function(data, ratio=0.1) {
  data[['noisy_data']] <- data$data %>%
    # add noise based on the sd of the col
    mutate(across(-data$meta_cols, ~ .x + rnorm(length(.x), 0, sd(.x) * ratio)))
  
  data[['noise_ratio']] <- ratio
  data
}



# ======== MODEL FITTING ========

# === ASCA ===

run_asca <- function(full_data, fact1, fact2) {
  noisy_full_data <- full_data
  
  data_design <- noisy_full_data %>% 
    select(c(fact1, fact2)) %>% 
    mutate(across(everything(), as_factor)) %>%  # convert all to factors
    as.matrix()
  
  data_for_asca <- noisy_full_data %>% 
    select(-c(fact1, fact2, "samp")) %>% 
    as.matrix()
  
  asca_res <- ASCA_decompose(
    d = data_design,
    x = data_for_asca,
    f = glue::glue("{fact1} + {fact2} + {fact1}:{fact2}")
    
  )
  
  fact_sing <- fact1
  fact_comb <- glue::glue('{fact1}:{fact2}')
  
  pca_on_asca <- (asca_res$decomposition[[fact_sing]] + asca_res$decomposition[[fact_comb]]) %>% prcomp()
  pca_on_asca
}


loadings_from_asca <- function(pca_res) {
  pca_res$rotation 
}


# === PCA ===

run_pca <- function(full_data, data_cols) {
  prcomp(full_data[data_cols])
}

loadings_from_pca <- function(pca_res) {
  pca_res$rotation
}


# === PLS ===

run_pls <- function(full_data, meta_cols, feat) {
  X <- full_data[setdiff(setdiff(colnames(full_data), c(feat)), meta_cols)]
  Y <- full_data[, feat]
  
  mixOmics::pls(X, Y) # run the method
  
}

components_from_pls <- function(pls_res) {
  rbind(pls_res$loadings$X, pls_res$loadings$Y)
}


importances_from_pls <- function(pls_res) {
  vip_scores <- mixOmics::vip(pls_res) %>%
    as_tibble(rownames="feat") %>% 
    mutate(
      comp1 = (comp1 - min(comp1)) / (max(comp1) - min(comp1)),
      comp2 = (comp2 - min(comp2)) / (max(comp2) - min(comp2))
    ) %>% 
    column_to_rownames('feat') %>% 
    bind_rows(pls_res$loadings$Y %>% as.data.frame()) %>% 
    as.matrix()
}



# ========== RANKINGS ========== 
get_distances <- function(values, type='euclidean', absolute=F, columns=NULL) {
  # columns can be null or a numeric vector
  if (is.null(columns))
    # use all the columns if null
    columns = 1:length(colnames(values))
  
  if (absolute)
    values <- abs(values)
  
  dists <- values  # default to using the values
  
  if (type != 'none' | is.null(type)) {
    # none means it just uses the values directly
    my_matrix <- values[, columns]  # use the first 2 components
    dist_matrix <- proxy::dist(my_matrix, method = type)
    
    # convert -> matrix -> df
    row_names <- rownames(my_matrix)
    dist_df <- as.data.frame(as.matrix(dist_matrix))
    dist_df$Row <- rownames(dist_df)
  }
  else {
    dist_df <-  as.data.frame(values[, columns])
  }
  dist_df
}

get_ranking <- function(values, type='euclidean', absolute=F, columns=NULL) {
  dist_df <- get_distances(values, type, absolute, columns=columns)
  row_names <- rownames(dist_df)
  
  rank_distances <- function(row_name) {
    rank(dist_df[row_name], ties.method = "average")
  }
  rankings_list <- lapply(row_names, rank_distances)
  rankings_df <- as.data.frame(do.call(rbind, rankings_list))
  
  rownames(rankings_df) <- row_names
  colnames(rankings_df) <- row_names
  
  # pivot to longer form 
  rank_df <- rankings_df  %>% 
    rownames_to_column('from') %>% 
    as_tibble() %>% 
    # select(-Row) %>% 
    pivot_longer(cols = -from, names_to = 'to', values_to = 'rank') %>% 
    
    # join with the long form distances
    inner_join(
      dist_df %>% 
        rownames_to_column('from') %>% 
        as_tibble() %>% 
        select(-Row) %>% 
        pivot_longer(cols = -from, names_to = 'to', values_to = 'dist') ,
      by = c('to', 'from')
    ) %>% 
    arrange(from, rank)
}


# special case for PLS
get_ranking_for_pls <- function(full_data, meta_cols, specs, type = 'euclidean', absolute = F) {
  # assumes last feature is the last in the pathway
  # fit an individual PLS model for each and compute the distance for each
  values <- specs %>% 
    map(~ importances_from_pls(run_pls(full_data, meta_cols, .x %>% pluck(-1)))) %>% 
    imap(~ get_distances(.x, type = type, absolute = absolute)[pluck(specs[[.y]], -1), , drop=F]) %>% 
    list_rbind() 
  
  print(values)
  
  # create the ranking as usual
  dist_df <- values
  row_names <- rownames(dist_df)
  col_names <- colnames(dist_df)
  
  rank_distances <- function(row_name) {
    rank(dist_df[row_name, ], ties.method = "average")
  }
  rankings_list <- lapply(row_names, rank_distances)
  rankings_df <- as.data.frame(do.call(rbind, rankings_list))
  
  rownames(rankings_df) <- row_names
  colnames(rankings_df) <- col_names
  
  
  # pivot to longer form 
  rank_df <- rankings_df  %>% 
    rownames_to_column('from') %>% 
    as_tibble() %>% 
    # select(-Row) %>% 
    pivot_longer(cols = -from, names_to = 'to', values_to = 'rank') %>% 
    
    # join with the long form distances
    inner_join(
      dist_df %>% 
        rownames_to_column('from') %>% 
        as_tibble() %>% 
        select(-Row) %>% 
        pivot_longer(cols = -from, names_to = 'to', values_to = 'dist') ,
      by = c('to', 'from')
    ) %>% 
    arrange(from, rank)
}

# get mapping from colname to type
cln_data_to_col_map <- function(clean_data) {
  tibble(
    type=clean_data$col_types,
    feature=setdiff(colnames(clean_data$data), clean_data$meta_cols)
  )
}


# filter and annotate rank results
filter_ranks <- function(ranks, col_map, specs) {
  # get the pairs of relevant comparisons
  filt_pairs <- specs %>% 
    imap(~ tibble(pathway = .y, main_feat = .x %>% pluck(-1), other_feat = .x)) %>%  
    list_rbind()
  
  # add the faeture type
  filt_pairs <- filt_pairs %>% 
    inner_join(col_map, by = c('other_feat' = 'feature')) %>% 
    # add pathway type metadata - whether it is a host or microbe only pathway, or comined
    mutate(
      pathway_has_host = as.numeric(str_extract(pathway, "^[^(]+\\((\\d+)\\)", 1)) > 0,
      pathway_has_microbe = as.numeric(str_extract(pathway, '^[^(]+\\([^(]+\\((\\d+)\\)', 1)) > 0
    )
  # join with the distances and drop the identity one
  ranks %>%  
    inner_join(filt_pairs, by=c('from' = 'main_feat', 'to' = 'other_feat'))  %>% 
    filter(from != to)
}



# ========== FULL PIPELINES ========== 

# these return results for different pathways and their components
# some do both

PCA_pipeline <- function(raw_data, clean_data) {
  specs <- raw_data$specs  # get specs from raw data
  
  # run PCA pipeline
  loadings <- loadings_from_pca(run_pca(clean_data$noisy_data, clean_data$data_cols))
  
  cross2(c('euclidean', 'cosine'), c(T, F)) %>% 
    map(~ {
      ranks <- get_ranking(loadings, type = .x[[1]], absolute = .x[[2]], columns=1:2)
      col_map <- cln_data_to_col_map(clean_data)
      filter_ranks(ranks, col_map, specs) %>% 
        mutate(dist_type = .x[[1]], dist_abs = .x[[2]])  # add the metadata for the run
    }) %>% 
    list_rbind() %>% 
    mutate(model_type = "PCA")
}


ASCA_pipeline <- function(raw_data, clean_data) {
  specs <- raw_data$specs  # get specs from raw data
  
  # get the factors
  fact1 <- clean_data$meta_cols %>%  pluck(-2)
  fact2 <- clean_data$meta_cols %>%  pluck(-1)

  # TODO -wrokaround
  # fact1 <- "EX_o2_m_lvl"
  # fact2 <- "EX_nh4_m_lvl"
  
  # run the asca pipline
  asca_res <- run_asca(clean_data$noisy_data, fact1, fact2)
  loadings <- loadings_from_asca(asca_res)
  
  cross2(c('euclidean', 'cosine'), c(T, F)) %>% 
    map(~ {
      ranks <- get_ranking(loadings, type = .x[[1]], absolute = .x[[2]], columns=1:2)
      col_map <- cln_data_to_col_map(clean_data)
      filter_ranks(ranks, col_map, specs) %>% 
        mutate(dist_type = .x[[1]], dist_abs = .x[[2]])  # add the metadata for the run
    }) %>% 
    list_rbind() %>% 
    mutate(model_type = "ASCA")
}


PLS_pipeline <- function(raw_data, clean_data) {
  specs <- raw_data$specs  # get specs from raw data
  
  col_map <- cln_data_to_col_map(clean_data)  # get col mapper
  
  # get the ranks and just return
  get_ranking_for_pls(clean_data$noisy_data, clean_data$meta_cols, raw_data$specs) %>% 
    filter_ranks(col_map, specs) %>% 
    mutate(model_type = "PLS")
}

dummy_pipeline <- function(raw_data, clean_data) {
  specs <- raw_data$specs  # get specs from raw data
  
  # just use the noisy data as lodings
  loadings <- clean_data$noisy_data %>% 
    select(-clean_data$meta_cols) %>% 
    as.matrix() %>% 
    t()
  
  cross2(c('euclidean', 'cosine'), c(T, F)) %>% 
    map(~ {
      ranks <- get_ranking(loadings, type = .x[[1]], absolute = .x[[2]], columns=NULL)
      col_map <- cln_data_to_col_map(clean_data)
      filter_ranks(ranks, col_map, specs) %>% 
        mutate(dist_type = .x[[1]], dist_abs = .x[[2]])  # add the metadata for the run
    }) %>% 
    list_rbind() %>% 
    mutate(model_type = "dummy")
}


run_MOFA <- function(clean_data, grouped=F) {
  # library(MOFA2)
  
  # given a clean data object
  # wrangle to correct format required by the MOFA2 library
  mofa_data <- clean_data$noisy_data %>% 
    mutate(samp = 1:length(samp)) %>% 
    pivot_longer(-c(clean_data$meta_cols), values_to = "value", names_to="feature") %>% 
    mutate(view = clean_data$col_types[match(feature, colnames(clean_data$noisy_data)) - length(clean_data$meta_cols)]) 
  
  # add the group comlumn or not
  if(!grouped)
    mofa_data <- mofa_data %>% 
      transmute(sample = as.character(samp), feature, view, value) 
  else
    mofa_data <- mofa_data %>% 
      # create a group columns to pass to mofa(exp conditions - samp)
      unite("group", !!!setdiff(clean_data$meta_cols, "samp")) %>%  
      transmute(sample = as.character(samp), feature, view, value, group)   
  
  # create MOFA object from the data
  mofa_object <- MOFA2::create_mofa_from_df(mofa_data, T)
  
  # use the default settings (gaussian kernels, scaling and all)
  data_opts <- MOFA2::get_default_data_options(mofa_object)
  model_opts <- MOFA2::get_default_model_options(mofa_object)
  train_opts <- MOFA2::get_default_training_options(mofa_object)
  train_opts$verbose <- F
  model_opts$num_factors <- 4
  
  # add the options to the object
  mofa_object <- MOFA2::prepare_mofa(
    object = mofa_object,
    data_options = data_opts,
    model_options = model_opts,
    training_options = train_opts
  )
  mofa_object@training_options$verbose <- F
  
  # run the training (assume mofapy2 is in the reticulate env)
  MOFA2::run_mofa(mofa_object, use_basilisk = F, save_data = F)
}


loadings_from_MOFA <- function(mofa_res) {
  # given a MOFA result, return the weights for each factor and features
  MOFA2::get_weights(mofa_res, scale=T) %>% 
    map(~ as.data.frame(.x) %>%  rownames_to_column("feat") %>%  as_tibble()) %>% 
    list_rbind() %>% 
    column_to_rownames("feat") %>% 
    as.matrix()
}


MOFA_pipeline <- function(raw_data, clean_data, grouped=F) {
  specs <- raw_data$specs  # get specs from raw data
  
  # run the mofa fitting and get the loadings from it
  mofa_res <- run_MOFA(clean_data, grouped = grouped)
  loadings <- loadings_from_MOFA(mofa_res)
  
  # run mofa in grouped mode or not
  mofa_type <- "MOFA"
  if(grouped)
    mofa_type <- "MOFA_GROUPED"
  
  # compute the possible distance metrics
  cross2(c('euclidean', 'cosine'), c(T, F)) %>% 
    map(~ {
      ranks <- get_ranking(loadings, type = .x[[1]], absolute = .x[[2]])
      col_map <- cln_data_to_col_map(clean_data)
      filter_ranks(ranks, col_map, specs) %>% 
        mutate(dist_type = .x[[1]], dist_abs = .x[[2]])  # add the metadata for the run
    }) %>% 
    list_rbind() %>% 
    mutate(model_type = mofa_type)
}


# ================ HIGH LEVEL PIPELINES ================


all_pipelines <- function(raw_data, clean_data, kinds = c('PCA', 'dummy')) {
  # TODO: add parameter to specify methods
  all_res <- list()
  
  # brute force if 
  if ('PCA' %in% kinds)
    all_res[['PCA']] <- PCA_pipeline(raw_data, clean_data)
  if ('ASCA' %in% kinds)
    all_res[['ASCA']] <- ASCA_pipeline(raw_data, clean_data)
  if ('PLS' %in% kinds)
    all_res[['PLS']] <- PLS_pipeline(raw_data, clean_data)
  if ('MOFA' %in% kinds)
    all_res[['MOFA']] <- MOFA_pipeline(raw_data, clean_data, grouped = F)
  if ('MOFA_GROUPED' %in% kinds)
    all_res[['MOFA_GROUPED']] <- MOFA_pipeline(raw_data, clean_data, grouped = T)
  if ('dummy' %in% kinds)
    all_res[['dummy']] <- dummy_pipeline(raw_data, clean_data)
  
  bind_rows(all_res)
}


run_pipeline_multi_noise <- function(raw_data, clean_data, 
                                     pipeline_func = all_pipelines, 
                                     noise_lvls = c(0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25),
                                     .progress = F) {
  
  # allow both formula and functions to be passed
  pipeline_func <- rlang::as_function(pipeline_func)
  
  map(noise_lvls, ~ {
    noise <- .x
    new_clean_data <- add_noise(clean_data, ratio = noise)
    
    pipeline_func(raw_data, new_clean_data) %>% 
      mutate(noise_lvl = noise)  # add a noise column
  }, .progress = .progress) %>% 
    bind_rows()
}


run_pipeline_mutli_sets <- function(
    files,  # list of files
    pipeline_func = all_pipelines,
    .progress = F,
    output_pattern = NULL,
    keep_max = 10,
    random_spec = c(F), # randomize all but the target metabolite in the spec
    seed = 42,
    keep_features = c("bm", "metab", "gene", "internal")
) {
  # allow both formula and functions to be passed
  pipeline_func <- rlang::as_function(pipeline_func)
  
  map_func <- furrr::future_map
  if (!is.null(output_pattern)) {
    # do not bother sending results back 
    map_func <- furrr::future_walk  
  }
  
  with_progress({
    prog <- progressor(steps = length(files))
    
    final_res <- map_func(
      files, 
      # set seed
      .options = furrr::furrr_options(seed = seed),
      ~ {
        filename <- .x  # to avoid name collision with .x
        
        # get the name of the dataset
        name <- load_data(filename) 
        name <- name$dataset_meta$dataset_full_name %>% 
          pluck(1)
        # will repeat inside
        print(name)
        
        # map over all values of random_spec (T or F)
        result <- unique(c(random_spec)) %>% map( ~ {
          print(.x)
          # load the raw data
          raw_data <- load_data(filename)
          # print(raw_data$dataset_meta)
          # print(name)
          
          # clean the data
          clean_data <- clean_up_data(raw_data)
          # subset features
          clean_data <-  subset_features(clean_data, keep_features, 
                                         imp_feats = clean_data$specs, keep_max = keep_max)
          clean_data <- add_noise(clean_data, ratio = 0.0) # add noise just to create the noisy_data df
          
          # create random specs if requested
          if(.x) {
            new_spec <- colnames(clean_data$noisy_data) %>% 
              # remove meta cols and existing specs
              setdiff(clean_data$meta_cols) %>% 
              setdiff(flatten_chr(clean_data$specs)) %>% 
              
              # sample 1 less than pathway length
              sample(length(intersect(flatten_chr(clean_data$specs), colnames(clean_data$noisy_data))) - 1) %>% 
              
              # append target feature last
              c(., flatten_chr(clean_data$specs) %>%  pluck(-1)) %>% 
              list("pathway" = .)
            
            # replace the values
            clean_data$spec <- new_spec
            raw_data$spec <- new_spec
          }
          
          # skip if file exists
          if(!is.null(output_pattern)) 
            if(file.exists(glue::glue(output_pattern))) {
              print('SKIP')
              return()
            }
          
          # return the result of the inner pipeline func
          pipeline_func(raw_data, clean_data) %>% 
            bind_cols(raw_data$dataset_meta) %>%  # add the dataset metadata
            mutate(random_spec = .x)  # flag to see if it was random or not
        }) %>% bind_rows()
        
        prog()
        
        if(!is.null(output_pattern)) {
          result %>% 
            write_parquet(glue::glue(output_pattern),  # output directly 
                          compression = "gzip", compression_level = 5)
          return(NULL)
        } 
        else {
          return(result)
        }
    }) 
    if(is.null(output_pattern) & !is.null(final_res)) {
      return(final_res %>% bind_rows())
    }
  })
}



# wrap a pipeline func for bootstrapping
run_pipeline_bootstrap <- function(
    raw_data, clean_data, 
    pipeline_func = all_pipelines, 
    n_rep = 30,
    n_samps = 10,
    .progress = F) {
  # subsamples noisy data
  # should be passed already noisy data so probably within multi-noise
  
  # allow both formula and functions to be passed
  pipeline_func <- rlang::as_function(pipeline_func)
  
  # run the number of replicates and samples
  map(as.vector(n_samps), \(n_samp) {
    map(1:n_rep, ~ {
      new_clean_data <- rlang::duplicate(clean_data)
      
      # get a set of samples
      selected_samps <- new_clean_data$data %>% 
        distinct(samp) %>% 
        slice_sample(n = n_samp, replace = T) %>% 
        pull(samp)
      
      # subsample
      new_clean_data$data <- new_clean_data$data %>% 
        # for each group subsample 
        filter(samp %in% selected_samps)
      
      # keep the same samples the regular data
      new_clean_data$noisy_data <- new_clean_data$noisy_data %>% 
        filter(samp %in% selected_samps)
      
      # then pass downstream
      pipeline_func(raw_data, new_clean_data) %>% 
        mutate(rep = .x, n_samps = n_samp)  # add the rep column
    }, .progress = .progress) %>% 
      bind_rows()
  }) %>% 
    bind_rows()
  
}
