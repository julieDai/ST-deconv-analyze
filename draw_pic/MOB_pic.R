library(CARD)
# 使用 read.csv() 函数加载 CSV 文件
Proportion <- read.csv("card/predictor_matrix_ref_MOB12*10.csv", header = TRUE, sep = ",",row.names = 1)
location <- read.csv("card/location_12.csv", header = TRUE, sep = ",",row.names = 1)
colors = c("#D7A4B2","#FFD2C5","#6489A1","#FF7146","#A7AF7A","#FFD92F","#4DAF4A","#D9D9D9","#7FC97F")


Proportion <- read.csv("card/predictor_matrix_ref_02dataset.csv", header = TRUE, sep = ",",row.names = 1)
location <- read.csv("card/location.csv", header = TRUE, sep = ",",row.names = 1)

## set the colors. Here, I just use the colors in the manuscript, if the color is not provided, the function will use default color in the package. 
colors = c("#FFD92F","#4DAF4A","#FCCDE5","#D9D9D9","#377EB8","#7FC97F","#BEAED4",
           "#FDC086","#FFFF99","#386CB0","#F0027F","#BF5B17","#666666","#1B9E77","#D95F02",
           "#7570B3","#E7298A","#66A61E","#E6AB02","#A6761D")
colors = c("#7C8080","#7570B3","#377EB8","#FFD2C5","#A7AF7A","#D9D9D9","#1B9E77","#BF5B17",
           "#D7A4B2","#FFD2C5","#6489A1","#FF7146","#A7AF7A","#FFD92F","#4DAF4A","#D9D9D9","#7FC97F",
           "#FDC086","#F0027F",
           "#E7298A","#E6AB02","#A6761D")

p1 <- CARD.visualize.pie(
  proportion = Proportion,
  spatial_location = location, 
  colors = colors, 
  radius = 0.52) ### You can choose radius = NULL or your own radius number
print(p1)


## select the cell type that we are interested
ct.visualize = c("GC","PGC","M.TC","OSNs","EPL.IN")
ct.visualize = c("Astrocytes","Neurons","Oligos","Vascular","Immune","Ependymal")


## visualize the spatial distribution of the cell type proportion
p2 <- CARD.visualize.prop(
  proportion = Proportion,        
  spatial_location = location, 
  ct.visualize = ct.visualize,                 ### selected cell types to visualize
  colors = c("lightblue","lightyellow","red"), ### if not provide, we will use the default colors
  #colors = viridis::viridis(256, option = "viridis"),
  NumCols = 7,                                 ### number of columns in the figure panel
  pointSize = 1.5)                             ### point size in ggplot2 scatterplot  
print(p2)

## visualize the spatial distribution of two cell types on the same plot
p3 = CARD.visualize.prop.2CT(
  proportion = Proportion,                             ### Cell type proportion estimated by CARD
  spatial_location = location,                      ### spatial location information
  ct2.visualize = c("GC","PGC"),              ### two cell types you want to visualize
  colors = list(c("lightblue","lightyellow","red"),c("lightblue","lightyellow","black")))       ### two color scales                             
print(p3)

p4 <- CARD.visualize.Cor(Proportion,colors = NULL) # if not provide, we will use the default colors
print(p4)

