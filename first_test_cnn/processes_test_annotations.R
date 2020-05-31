library(tidyverse)

# Get the most common images from my first quick run thru in classifying things
# Make into a nice csv of the  image filename + labels

annotations = read_csv('imageant_stuff/simple_annotation_test.csv') %>%
  select(-time)

categories = read_csv('classification_categories.csv') %>%
  select(class_category, class_subcategory, class_id, class_description)

# combine the crop/pasture field status subcategories into one
annotations = annotations %>%
  pivot_longer(cols = c('field_status_crop','crop_type','snow_present','field_status_pasture','field_flooded'),
               names_to = 'class_subcategory', values_to = 'class_id') %>%
  filter(!is.na(class_id)) %>%
  left_join(categories, by=c('class_subcategory','class_id')) %>%
  select(-class_description, -class_subcategory) %>%
  pivot_wider(names_from = 'class_category', values_from = 'class_id')

classes_to_keep =  annotations %>%
  count(crop_type, field_status,snow_present,field_flooded) %>% 
  filter(n>=15) %>%
  mutate(keep='yes') %>%
  select(-n)

# make a classification for each images in the subset to keep
# combine the crop/pasture field status into one category
image_classifications = annotations %>%
  left_join(classes_to_keep, by=c('crop_type','field_status','snow_present','field_flooded')) %>%
  filter(keep=='yes') %>%
  select(-keep, -field_flooded)

write_csv(image_classifications,'first_test_cnn/image_classifications.csv')
