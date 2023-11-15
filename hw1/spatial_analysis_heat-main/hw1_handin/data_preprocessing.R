# Code to conduct spatial random-effect meta analysis to data applications.
library(parallel)
library(USAboundaries)
library(meta)
library(metafor)
library(dplyr)
library(sp)
library("raster")
library("sf")
library(rgeos)
library(lubridate)
library(sf)

ts <- readRDS("ts_heatindex_heatwarnings_byFIPS_2006_2016.rds")

heat_us <- readRDS("hotspot_results.RDS")


ts$weekday <- wday(ts$Date, week_start=1)
ts$weekday <- ifelse(ts$weekday<6, 1, 0)
states <- st_read("cb_2013_us_county_500k/cb_2013_us_county_500k.shp")
# Convert the StateCounty_FIPS column to a fixed width character with leading zeros
ts$StateCounty_FIPS <- sprintf("%05s", as.character(ts$StateCounty_FIPS))

ts_geo <- as.data.frame(ts_geo)
# Perform the right join
ts_geo <- ts %>% right_join(states, by = c("StateCounty_FIPS" = "GEOID"))

lat_long <- as.data.frame(st_coordinates(st_centroid(st_geometry(states))))
lat_long <- cbind(lat_long, states)
ts_geo <- ts %>% right_join(lat_long, by = c("StateCounty_FIPS" = "GEOID"))


saveRDS(ts_geo, file="ts_geo.RDS")

# Read the data from the .RDS file
df_read <- readRDS("ts_geo.RDS")

# Print the data
print(df_read)
