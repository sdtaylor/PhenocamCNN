library(tidyverse)
library(phenocamr)

phenocam_sites = phenocamr::list_rois() %>%
  select(phenocam_name = site, lat, lon, roi_type=veg_type, roi_id =roi_id_number, first_date, last_date, site_years) 

phenocam_sites = phenocam_sites %>%
  filter(roi_type == 'AG',
         site_years > 2) %>%
  filter((roi_id %% 1000) == 0) # No experimental ROIs, which usually end in X001

write_csv(phenocam_sites, 'site_list.csv')
