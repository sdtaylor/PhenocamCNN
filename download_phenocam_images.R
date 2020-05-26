library(tidyverse)

images_for_download = read_csv('images_for_download.csv')

base_url = 'https://phenocam.sr.unh.edu/data/archive/'
dest_folder= './data/phenocam_images/'

failed_downloads = tibble(phenocam_name=c(),date=c(),image_filename=c())

for(image_i in 1:nrow(images_for_download)){
  phenocam_name = images_for_download$phenocam_name[image_i]
  
  year = lubridate::year(images_for_download$date[image_i])
  
  month = lubridate::month(images_for_download$date[image_i])
  month = ifelse(month<10, paste0('0',month), as.character(month))
  
  image_filename = images_for_download$image_filename[image_i]
  
  download_url = paste0(base_url,phenocam_name,'/',year,'/',month,'/',image_filename)
  dest_path    = paste0(dest_folder,image_filename)
  
  download_attempt = try(download.file(url = download_url,destfile = dest_path,))
  
  if(class(download_attempt) == 'try-error'){
    failed_downloads = failed_downloads %>%
      add_row(phenocam_name = phenocam_name,
              date          = images_for_download$date[image_i],
              image_filename = image_filename)
  }
  
}
