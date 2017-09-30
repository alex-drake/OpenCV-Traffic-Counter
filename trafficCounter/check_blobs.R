## Read in csvs and match data from tracked blobs in
## vehicle detection process

library(dplyr)
library(ggplot2)

setwd("~/Veh Detection/Sample Scripts")

# read in blobs data
blobs <- read.csv("/Users/datascience9/Veh Detection/Sample Scripts/tracked_blobs.csv",
                  header = T, stringsAsFactors = F)
# read in contours data
contours <- read.csv("/Users/datascience9/Veh Detection/Sample Scripts/tracked_conts.csv",
                     header = T, stringsAsFactors = F)
blobs$id <- as.character(blobs$id)

# do some plots
cPlot <- ggplot(contours, aes(x = width, y = height)) + geom_point() +
  xlim(0,100) + ylim(0,100)
bPlot <- ggplot(blobs, aes(x = vector_y, y = vector_x, color=id)) + geom_point() +
  scale_color_discrete() +
  ylim(0,mean(blobs$vector_x, na.rm = T)) + xlim(-180,180)

# store plots
# dev.copy(png, "../Outputs/contourPlot.png")
# cPlot
# dev.off()
# 
# dev.copy(png, "../Outputs/blobPlot.png")
# bPlot
# dev.off()
ggsave("../Outputs/contourPlot.png",cPlot)
ggsave("../Outputs/blobPlot.png",bPlot)