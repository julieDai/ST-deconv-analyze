library(CARD)
# 使用 read.csv() 函数加载 CSV 文件
Proportion <- read.csv("data_draw/AE_ttest_24/predictor_matrix_ref_24.csv", header = TRUE, sep = ",",row.names = 1)
location <- read.csv("data_draw/AE_ttest_24/pre_location.csv", header = TRUE, sep = ",",row.names = 1)

## set the colors. Here, I just use the colors in the manuscript, if the color is not provided, the function will use default color in the package. 
colors = c("#FFD92F","#4DAF4A","#FCCDE5","#377EB8","#D9D9D9","#7FC97F","#BEAED4",
           "#FDC086","#FFFF99","#386CB0","#F0027F","#BF5B17","#666666","#1B9E77","#D95F02",
           "#7570B3","#E7298A","#66A61E","#E6AB02","#A6761D")
p1 <- CARD.visualize.pie(
  proportion = Proportion,        
  spatial_location = location, 
  colors = colors, 
  radius = 0.52) ### You can choose radius = NULL or your own radius number
print(p1)

## select the cell type that we are interested
ct.visualize = c("Acinar_cells","Cancer_clone_A","Cancer_clone_B","Ductal_terminal_ductal_like","Ductal_CRISP3_high-centroacinar_like","Ductal_MHC_Class_II","Ductal_APOL1_high-hypoxic","Fibroblasts")

## visualize the spatial distribution of two cell types on the same plot
p3 = CARD.visualize.prop.2CT(
  proportion = Proportion,                             ### Cell type proportion estimated by CARD
  spatial_location = location,                      ### spatial location information
  ct2.visualize = c("Neurons","Oligos"),              ### two cell types you want to visualize
  colors = list(c("lightblue","lightyellow","red"),c("lightblue","lightyellow","black")))       ### two color scales                             
print(p3)

p4 <- CARD.visualize.Cor(Proportion,colors = NULL) # if not provide, we will use the default colors
print(p4)
