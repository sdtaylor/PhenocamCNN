

This repo is now at [https://github.com/sdtaylor/PhenocamCNN2/](https://github.com/sdtaylor/PhenocamCNN2/)

## File list
Descriptions of scripts in the repo in the order they're  meant to be run
1. generate_site_list.R - Download the latest phenocam metadata and choose appropriate sites, creating site_list.csv
2. download_phenocam_data.R - Download 3day gcc data for everything in site_list.csv into data/phenocam_gcc/
3. generate_annoation_file_list.R - From the daily gcc data determine which images to fetch for annotation/cnn training, generating image_annotation_list.R
4. download_phenocam_images.R   - download everything in images_for_download.csv to data/phencam_images/
