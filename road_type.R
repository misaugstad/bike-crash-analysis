library(plyr)

road.data<- read.csv(file="Hello_OSM.csv", header=TRUE, sep=",")
road.data$type <- revalue(road.data$type,
                             c("bridleway"="1", "construction"="2", "cycleway"="3", "disused"="4",
                               "elevator"="5", "escalator"="6", "footway"="7", "living_street"="8",
                               "motorway"="9", "motorway_link"="10", "path"="11", "pedestrian"="12",
                               "primary"="13", "primary_link"="14", "private"="15", "proposed"="16",
                               "residential"="17", "secondary"="18", "secondary_link"="19",
                               "service"="20", "steps"="21", "tertiary"="22", "tertiary_link"="23",
                               "track"="24", "trunk"="25", "trunk_link"="26", "unclassified"="27"))

write.csv(road.data, file= "~/Dropbox/Course@UMD/INFM750/Project/R scripts/Hello_OSM_NUM.csv")

