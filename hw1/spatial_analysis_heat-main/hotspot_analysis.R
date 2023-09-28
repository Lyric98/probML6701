library(tidyverse)
library(sf)
library(tigris)
library(ggplot2)
library(spdep)
library(Guerry)

# data load in
stochastic_intervention_fips<-readRDS("stochastic_intervention_fips.RDS") %>% 
  mutate(fips=as.character(fips)) %>% 
  mutate(fips=str_pad(fips, width = 5,pad = "0")) %>% select(fips,Estimate)

# Load counties shapefile data for the entire US
options(tigris_class = "sf")
counties <- tigris::counties(cb = TRUE)

# Combine STATEFP and COUNTYFP to create a full FIPS code
counties$GEOID <- paste0(counties$STATEFP, counties$COUNTYFP)

# merge the stochastic_intervention_fips with the shapefile
merged_data <- left_join(counties, stochastic_intervention_fips, by = c("GEOID" = "fips")) %>% 
  drop_na()

map_1<-ggplot() +
  geom_sf(data = merged_data, 
          aes(fill = Estimate), 
          color = NA) +
  scale_fill_viridis_c(option = "viridis") +
  theme_minimal() +
  theme(axis.text = element_blank(), 
        axis.ticks = element_blank(), 
        panel.grid = element_blank())

ggsave("map_1_SIF.png",map_1)

## spatial clustering
# Convert to SpatialDataFrame for spdep functions
merged_data_sp <- as(merged_data, "Spatial")

# Define spatial weights matrix
nb <- poly2nb(merged_data_sp)

#2 regions with no links:1071 1181
# Remove regions with no neighbors from the data and the neighbors list
merged_data_sp <- merged_data_sp[-c(1071,1181), ]
merged_data_1<-merged_data[-c(1071,1181),]

nb <- poly2nb(merged_data_sp)
# Define spatial weights matrix
listw <- nb2listw(nb)

# do the localmoran test
hotspot_result <- localmoran(merged_data_sp$Estimate, listw)

saveRDS(hotspot_result, "hotspot_results.rds")

##Local Moran's I is a statistic used to identify local spatial autocorrelation in data. 
##While global Moran's I provides a measure of overall spatial autocorrelation, the Local Moran's I focuses on identifying where clusters or outliers are in the spatial dataset.

####Clusters: These are areas where similar values are grouped together. This can be high-high (high values surrounded by high values) or low-low clusters (low values surrounded by low values).

####Outliers: These are areas where values are dissimilar from their surrounding neighbors. This can be high-low (high values surrounded by low values) or low-high clusters (low values surrounded by high values).

## visualize the hotspot results

## For a positive Z.Ii and a small p-value (<=0.05), you have a hotspot.
## For a negative Z.Ii and a small p-value (<=0.05), you have a significant coldspot or spatial outlier.
merged_data_sp$Pr_z<-hotspot_result[,"Pr(z != E(Ii))"]
merged_data_sp$Z.Ii<-hotspot_result[,"Z.Ii"]
merged_data_sf<-st_as_sf(merged_data_sp)

## draw the map// at significant level =0.05
merged_data_sf %>% 
  mutate(
    classification=case_when(
      Z.Ii>0&Pr_z<=0.05~"Hotspot",
      Z.Ii<0&Pr_z<=0.05~"Coldspot",
      TRUE ~ "Insignificant"
    ),
    classification=factor(classification,levels=c("Hotspot","Coldspot","Insignificant"))
  ) %>% 
  ggplot() +
  geom_sf(aes(fill = classification)) +
  scale_fill_manual(values = c("Hotspot" = "red", "Coldspot" = "blue", "Insignificant" = "gray")) +
  theme_minimal() +
  labs(fill = "Significance",
       title = "Hotspot Analysis") 



