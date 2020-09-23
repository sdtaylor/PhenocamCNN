library(tidyverse)

"
This takes the output from a keras image classificaiton model and prepares it for post-processing
in a hidden markov model (HMM). 
"

classes = read_csv('classification_categories_crop_only.csv') %>%
  rename(class = class_description)

image_predictions = read_csv('results/vgg16_v2_predictions.csv') %>%
  select(-filepath)

image_predictions = image_predictions %>%
  mutate(file = str_remove(file, '.jpg')) %>%
  separate(file, c('phenocam_name','datetime'), '_', extra='merge') %>%
  mutate(datetime = str_replace_all(datetime,'_','-')) %>%
  mutate(datetime = lubridate::as_datetime(datetime, format='%Y-%m-%d-%H%M%S')) %>%
  mutate(hour = lubridate::hour(datetime),
         date = lubridate::date(datetime)) 

states =       c("emergence","growth state","tassles/flowering","senescing","fully senesced","harvested/plowed","snow","flooded","unknown")
image_predictions = image_predictions %>%
  filter( hour %in% c(11,12,13,14)) %>%
  pivot_longer(cols=all_of(states), names_to = 'class', values_to = 'probability') %>%
  group_by(phenocam_name, date) %>%
  slice_max(probability, n=1) %>%
  slice_head(n=1) %>%  # When there multiple images/day with the same high probability (ie. multiple images of snow in 1 day all 
                       # with probability = 1.0), just pick the 1st one.
  ungroup() 

group_date_sequences = function(df){
  full_date_range = tibble(date = seq(min(df$date), max(df$date), 'day'))
  
  df = full_join(df, full_date_range, by='date') %>%
    arrange(date)
  ########################
  ########################
  # Count the size of daily sequences which have an available image file 
  # for all days. And label each sequence with a unique id
  df$running_tally = NA
  df$site_sequence_id = NA
  present_tally=0
  seq_id=0
  for(i in 1:nrow(df)){
    
    current_row_present = !is.na(df$class[i])
    if(current_row_present){
      present_tally = present_tally + 1
      missing_tally = 0
    } else {
      present_tally = 0
    }
    
    if(present_tally==1){
      seq_id = seq_id+1
      df$site_sequence_id[i] = seq_id
    } else if(present_tally>1){
      df$site_sequence_id[i] = seq_id
    }
    
    df$running_tally[i] = present_tally
  }
  return(df)
}

###############################
# Group each sites timeseries into subsets, such that each subset has atleast 30 consecutive days.
# short subsets and any gaps are dropped for now.
image_predictions = image_predictions %>%
  group_by(phenocam_name) %>%
  do(group_date_sequences(.)) %>%
  ungroup()

image_predictions = image_predictions %>%
  group_by(phenocam_name, site_sequence_id) %>%
  filter(max(running_tally) >= 30)

image_predictions = image_predictions %>%
  left_join(classes, by='class')

write_csv(image_predictions, 'results/image_predictions_for_hmm.csv')
  