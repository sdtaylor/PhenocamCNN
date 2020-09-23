library(tidyverse)
library(ggnewscale)
library(patchwork)

################################


final_results = read_csv('data/final_results.csv')

classes = read_csv('classification_categories_crop_only.csv')


################################
# Bring in phenocam gcc daily data and transition dates


state_colors = c('springgreen1','green4',   'hotpink3',          'gold',     'darkorange4',  'tan3',             'grey50','blue','black')

ggplot(final_results, aes(x=date, y=gcc)) + 
  #geom_point(aes(color=as.factor(cnn_class_id))) + 
  geom_point(aes(y=gcc+0.25,color=as.factor(hmm_class_id))) + 
  scale_color_manual(values=state_colors, labels = classes$class_description) + 
  new_scale_color() + 
  geom_point(aes(y=0.8, x=date, color=gcc_based_stage)) + 
  scale_x_date(date_breaks = 'month') + 
  facet_wrap(~phenocam_name, ncol=1)

ggplot(final_results, aes(x=date, y=ec_gpp)) + 
  geom_point() + 
  geom_point(aes(y=25,color=as.factor(hmm_class_id))) + 
  scale_color_manual(values=state_colors, labels = classes$class_description) + 
  new_scale_color() + 
  geom_point(aes(y=20, x=date, color=gcc_based_stage)) + 
  scale_x_date(date_breaks = 'month') + 
  facet_wrap(~phenocam_name, ncol=1)

yearly_summaries = final_results %>%
  gather(class_type, class, cnn_class_id, hmm_class_id, gcc_based_stage) %>% 
  group_by(phenocam_name, year, crop, class_type, class) %>%
  summarise(total_gpp = sum(ec_gpp, na.rm=T),
            total_days = sum(!is.na(ec_gpp)),
            total_possible_days = n()) %>%
  ungroup() %>%
  filter(class_type == 'hmm_class_id') %>%
  filter(crop != 'None')

barplot_outline = ggplot(yearly_summaries, aes(x=paste0(phenocam_name,'\n',crop), fill=as.factor(class))) + 
  scale_fill_manual(values = c('#377eb8','#4daf4a','#e41a1c','#984ea3','#ff7f00'),breaks=classes$class_id, labels=classes$class_description) +
  theme_bw() + 
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        legend.position = 'none')

top_plot = barplot_outline + 
  geom_col(aes(y=total_gpp)) + 
  scale_y_continuous(limits=c(0,1550),expand = c(0,0)) + 
  labs(y='Total gC / m^2 over season')

middle_plot = barplot_outline + 
  geom_col(aes(y=total_gpp/total_days)) + 
  scale_y_continuous(limits = c(0,42), expand = c(0,0)) + 
  theme(legend.position = c(0.5,0.7),
        legend.background = element_rect(color='black'),
        legend.title = element_text(size=10)) +
  labs(y='Average gC / m^2 / day^-1', fill='Growing Stage')

bottom_plot = barplot_outline + 
  geom_col(aes(y=total_days)) + 
  scale_y_continuous(limits = c(0,165), expand = c(0,0)) + 
  theme(axis.text.x = element_text()) +
  labs(y='Total duration (days)', fill='Growing Stage')

top_plot + middle_plot +  bottom_plot + plot_layout(ncol=1)
