library(tidycensus)
census_api_key("83c447da325b837becf3fa4b7020479b9d34ccab", install = TRUE)

counties_pop <- get_acs(geography = "county", 
                        variables = "B01003_001", 
                        geometry = FALSE)

county = 1
Sim2 = data.frame(logOR = seq(-2.3, 2.3, length.out = 50),
                  estimated = c(mean.est.eff[county,] - mean.est.eff[county, 25],
                                mean.est.eff[county,] - mean.est.eff[county, 25]), 
                  lower = c(mean.est.eff[county,] - mean.est.eff[county, 25] - tu_cs[county, 25],
                            mean.est.eff[county,] - mean.est.eff[county, 25] - 1.96*sd_bound[county, 25]),
                  upper = c(mean.est.eff[county,] - mean.est.eff[county, 25] + tu_cs[county, 25],
                            mean.est.eff[county,] - mean.est.eff[county, 25] + 1.96*sd_bound[county, 25]),
                  Methods = c(rep("Time-uniform CSs", 50), rep("Point-wise CIs", 50)))

plot(Sim2$logOR[25:50], (Sim2$estimated)[25:50], type = "n")

est.eff <- NULL
for (county in 1:N) {

Sim2 = data.frame(logOR = seq(-2.3, 2.3, length.out = 50),
                  estimated = c(mean.est.eff[county,] - mean.est.eff[county, 25],
                                mean.est.eff[county,] - mean.est.eff[county, 25]), 
                  lower = c(mean.est.eff[county,] - mean.est.eff[county, 25] - tu_cs[county, 25],
                            mean.est.eff[county,] - mean.est.eff[county, 25] - 1.96*sd_bound[county, 25]),
                  upper = c(mean.est.eff[county,] - mean.est.eff[county, 25] + tu_cs[county, 25],
                            mean.est.eff[county,] - mean.est.eff[county, 25] + 1.96*sd_bound[county, 25]),
                  Methods = c(rep("Time-uniform CSs", 50), rep("Point-wise CIs", 50)))

est.eff[county] <- Sim2$estimated[50] - Sim2$estimated[25]

}

aggregate_fips_heat <- data.frame(cbind(est = est.eff, 
                                        fips = sprintf("%05d", unique(ts_byFIPS$StateCounty_FIPS))))
aggregate_fips_heat$est <- as.numeric(aggregate_fips_heat$est)

aggregate_fips_heat <- merge(aggregate_fips_heat, counties_pop[,c(1,4)], by.x = "fips", by.y = "GEOID")

aggregate_fips_heat$rank = rank(-aggregate_fips_heat$est/aggregate_fips_heat$estimate)

heat_us <- mutate(aggregate_fips_heat,
                  STATEFP = str_sub(fips, 1, 2),
                  COUNTYFP = str_sub(fips, 3, 5))
str(heat_us)
str(states)
states$STATEFP=as.character(states$STATEFP)
states$COUNTYFP=as.character(states$COUNTYFP)
states_heat <- left_join(states, heat_us, by = c("STATEFP", "COUNTYFP"))

g1 <- ggplot(states_heat)+
  xlim(-125,-65)+ylim(25,50)+
  #  geom_sf(aes(fill = PD_p),color=NA,size=0.025)+
  geom_sf(aes(fill = rank),color='grey',size=0.005)+
  #  scale_fill_viridis_c(option="magma",begin=0.4)+
  scale_fill_gradient2(expression(paste("Effect Size")),low  = "#1e90ff", mid="#ffffba", high = "#8b0000",midpoint = median(heat_us$rank, na.rm = T),
                       breaks = quantile(heat_us$rank, c(0.25,0.50,0.75), na.rm = T),
                       limits = c(min(heat_us$rank,na.rm = T), max(heat_us$rank,na.rm = T)), 
                       na.value = "grey") +
  labs(title = expression(paste("Alert Effectiveness in All Counties 2006-2016"))) +
  theme_minimal() +
  theme(plot.title = element_text(size = 30*2,hjust = 0.5, vjust = -3),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        line = element_blank(),
        axis.title = element_blank(),
        legend.position = "bottom",
        legend.direction = "horizontal", 
        legend.text = element_text(angle = 60,  size = 20*2),
        legend.text.align = 0.75,
        legend.title = element_text(size = 18*2),
        legend.key.width = unit(150*2, "points"),
        panel.grid.major = element_line(colour = "transparent"))

jpeg("est.jpeg", height = 1024*0.6*2*5, width = 1024*2*5, res = 72*5)
g1
dev.off()
