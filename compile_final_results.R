library(tidyverse)
library(zoo)

# --------------------------------
# This brings together the final image classifications, the HMM post-processing
# the original GCC, and the eddy covariance derived GPP
# --------------------------------

tower_matching_info = read_csv('data/phenocam_ec_tower_matching.csv')

sites = c("arsmorris2","cafcookeastltar01","cafboydnorthltar01","mandani2","arsmorris1")

start_date = '2018-01-01'
end_date   = '2018-12-31'

full_date_range = seq(as.Date(start_date), as.Date(end_date), by='day')
full_date_df = expand_grid(date = full_date_range, phenocam_name = sites)

# --------------------------------
# Read in site EC data
# -----------------------------------

process_daily_ec_file = function(full_file_path){
  filename = basename(full_file_path)
  tower_name = str_split(filename, '_')[[1]][5]
  site_code = str_split(filename, '_')[[1]][4]
  
  read_csv(full_file_path) %>%
    mutate(TimeStamp = as.Date(TimeStamp, '%m/%d/%Y'),
           site_code = site_code,
           ec_tower_name = tower_name) %>%
    select(date = TimeStamp, site_code,ec_tower_name, ec_gpp = GPP_L3_Sum_Carbon) # gC m-2 day-1
}

ec_files = list.files('./data/ec_data/', pattern = 'EC_LTAR*', full.names = T)
ec_data = purrr::map_dfr(ec_files, process_daily_ec_file) %>%
  left_join(select(tower_matching_info,site_code,phenocam_name, ec_tower_name, location_id), by=c('site_code','ec_tower_name'))

ec_data = ec_data %>%
  select(phenocam_name, date, ec_gpp)

# -----------------------------------
# Bring in phenocam gcc daily data and transition dates
# -----------------------------------

site_regex = paste(sites, collapse = '|')
site_regex = paste0('(',site_regex,')')

read_gcc = function(f, type='gcc'){
  rows_to_skip = case_when(
    type == 'gcc' ~ 24,
    type == 'transition' ~ 16
  )
  phenocam_name = str_split(basename(f),'_')[[1]][1]
  df = read_csv(f, skip = rows_to_skip)
  df$phenocam_name = phenocam_name
  return(df)
}

all_gcc_files = list.files('./data/phenocam_gcc/',pattern = '1day.csv', full.names = TRUE) 
all_gcc_files = all_gcc_files[grepl(site_regex, all_gcc_files)]
all_gcc = purrr::map_df(all_gcc_files, read_gcc, type='gcc')

all_gcc = all_gcc %>%
  filter(phenocam_name %in% sites) %>%
  select(phenocam_name, date, gcc = smooth_gcc_90) 

# transition dates from the phenocam data

all_transition_files = list.files('./data/phenocam_gcc/', pattern = '1day_transition_dates.csv', full.names = TRUE)
all_transition_files = all_transition_files[grepl(site_regex, all_transition_files)]
all_transitions = purrr::map_df(all_transition_files, read_gcc, type='transition') %>%
  filter(gcc_value == 'gcc_90') %>%
  select(phenocam_name = site, direction, transition_10, transition_25, transition_50) %>%
  pivot_longer(cols= starts_with('transition'), names_to='threshold', values_to='date') 

# assign categories based on rising/falling trend
all_transitions = all_transitions %>%
  mutate(gcc_based_stage = case_when(
    direction == 'falling' & threshold == 'transition_50' ~ 'senescing',
    direction == 'falling' & threshold == 'transition_25' ~ 'senescing',
    direction == 'falling' & threshold == 'transition_10' ~ 'senesced',
    direction == 'rising'  & threshold == 'transition_50' ~ 'peak',
    direction == 'rising'  & threshold == 'transition_25' ~ 'growth',
    direction == 'rising'  & threshold == 'transition_10' ~ 'growth',
  ))

# fill in transition data so that every day has a gcc based  label
all_transitions = full_date_df %>%
  left_join(all_transitions, by=c('date','phenocam_name')) %>%
  arrange(phenocam_name, date) %>%
  group_by(phenocam_name) %>%
  mutate(gcc_based_stage = zoo::na.locf(gcc_based_stage, na.rm=FALSE)) %>%
  ungroup()

all_transitions = all_transitions %>%
  select(phenocam_name, date, gcc_based_stage)



# -----------------------------------
# image classifications
# -----------------------------------
hmm_predictions = read_csv('results/image_predictions_hmm_final.csv') %>%
  filter(phenocam_name %in% sites) %>%
  select(phenocam_name, date, cnn_class_id = class_id, hmm_class_id)

# -----------------------------------
# Crop types. I annotated these by hand to match up with classifier results.
# -----------------------------------
crop_type_info = tribble(
  ~phenocam_name,       ~date_start, ~date_end, ~crop,
  'arsmorris1',         '2018-05-28','2018-11-05', 'corn',
  'arsmorris2',         '2018-05-21','2018-08-04', 'wheat',
  'arsmorris2',         '2018-09-12','2018-11-05', 'cover crop',
  'cafboydnorthltar01', '2018-03-14','2018-04-29', 'cover crop', # def looks grassy, maybe just volunteers?
  'cafboydnorthltar01', '2018-05-01','2018-09-09', 'lentils?',  # gotta ask eric
  'cafcookeastltar01',  '2018-03-29','2018-05-01', 'cover crop', # another grassy thing
  'cafcookeastltar01',  '2018-05-20','2018-08-10', 'lentils?',
  'mandani2',           '2018-06-20','2018-10-01', 'soybean',
)

# expand to a crop label for all dates
crop_labels =  crop_type_info %>%
  group_by(phenocam_name, crop) %>%
  summarise(date = seq(as.Date(date_start), as.Date(date_end), by='day')) %>%
  ungroup() %>%
  right_join(full_date_df, by=c('phenocam_name','date')) %>%
  mutate(crop = replace_na(crop, 'None'))



# -----------------------------------
# combine everything
# -----------------------------------

all_data = hmm_predictions %>%
  left_join(ec_data, by=c('phenocam_name','date')) %>%
  left_join(all_gcc, by=c('phenocam_name','date')) %>%
  left_join(all_transitions, by=c('phenocam_name','date')) %>%
  left_join(crop_labels, by=c('phenocam_name','date'))

all_data = all_data %>%
  filter(date >= start_date, date <= end_date)

all_data = all_data %>%
  mutate(year = lubridate::year(date),
         doy  = lubridate::yday(date))

write_csv(all_data, 'data/final_results.csv')
