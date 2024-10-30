library(scales)

options(scipen=999)

# drawAurocHeatmap <- function(data, title = "Heatmap") {
# 
#   rownames(data) <- c("MDCR (8.1%)", "IqviaGer (3.4%)", "OptumSES (6.9%)", "OptumPanther (4.1%)", "CPRD (2.0)", "IPCI")
#   colnames(data) <- c("MDCR", "IqviaGer", "OptumSES", "OptumPanther", "CPRD", "IPCI")
# 
#   p <- heatmaply(data,
#             dendrogram = "none",
#             ylab = "Development database",
#             xlab = "Validation database",
#             main = title,
#             # cellnote = TRUE,
#             draw_cellnote = TRUE,
#             # limits = c(0.5, 1.0),
#             digits = 2L,
#             cellnote_textposition = "middle center",
#             scale = "none",
#             na.value = "grey50",
#             limits = c(0.5, 1.0),
#             # margins = c(60,100,NA,NA),
#             grid_color = "white",
#             grid_width = 0.00002,
#             titleX = TRUE,
#             hide_colorbar = FALSE,
#             branches_lwd = NULL,
#             label_names = c("Development", "Validation:", "AUROC"),
#             fontsize_row = 10,
#             fontsize_col = 10,
#             labCol = colnames(data),
#             labRow = rownames(data),
#             heatmap_layers = theme(axis.line=element_blank()))
#   print(p)
# }


drawAuprcHeatmap <- function(data, title = "Heatmap") {
  
  rownames(data) <- c("MDCR", "IqviaGer", "OptumSES", "OptumPanther", "IPCI")
  colnames(data) <- c("MDCR", "IqviaGer", "OptumSES", "OptumPanther", "IPCI")
  
  p <- heatmaply(data,
                 dendrogram = "none",
                 ylab = "Development database",
                 xlab = "Validation database", 
                 main = title,
                 # cellnote = TRUE,
                 draw_cellnote = TRUE,
                 # limits = c(0.5, 1.0),
                 digits = 2L,
                 cellnote_textposition = "middle center",
                 scale = "none",
                 na.value = "grey50",
                 limits = c(0.0, 1.0),
                 # margins = c(60,100,NA,NA),
                 grid_color = "white",
                 grid_width = 0.00002,
                 titleX = TRUE,
                 hide_colorbar = FALSE,
                 branches_lwd = NULL,
                 label_names = c("Development", "Validation:", "AUROC"),
                 fontsize_row = 10,
                 fontsize_col = 10,
                 cellnote_size = 12,
                 labCol = colnames(data),
                 labRow = rownames(data),
                 heatmap_layers = theme(axis.line=element_blank()))
  print(p)
}

drawAurocHeatmap <- function(data, title = "Heatmap") {
  
  rownames(data) <- c("MDCR", "IQGER", "OPSES", "OPEHR", "IPCI")
  colnames(data) <- c("MDCR", "IQGER", "OPSES", "OPEHR", "IPCI")
  
  p <- heatmaply(data,
                 dendrogram = "none",
                 ylab = "Development database",
                 xlab = "Validation database", 
                 main = title,
                 # cellnote = TRUE,
                 draw_cellnote = TRUE,
                 # limits = c(0.5, 1.0),
                 digits = 2L,
                 cellnote_textposition = "middle center",
                 scale = "none",
                 na.value = "grey50",
                 limits = c(0.5, 1.0),
                 # margins = c(60,100,NA,NA),
                 grid_color = "white",
                 grid_width = 0.00002,
                 titleX = TRUE,
                 hide_colorbar = FALSE,
                 branches_lwd = NULL,
                 label_names = c("Development", "Validation:", "AUROC"),
                 fontsize_row = 10,
                 fontsize_col = 10,
                 labCol = colnames(data),
                 labRow = rownames(data),
                 heatmap_layers = theme(axis.line=element_blank()))
  print(p)
}

drawCalibrationHeatmap <- function(data, title = "Heatmap") {
  
  rownames(data) <- c("MDCR", "IQGER", "OPSES", "OPEHR", "IPCI")
  colnames(data) <- c("MDCR", "IQGER", "OPSES", "OPEHR", "IPCI")
  
  p <- heatmaply(data,
                 dendrogram = "none",
                 ylab = "Development database",
                 xlab = "Validation database", 
                 main = title,
                 colors = viridis(n = 256, alpha = 1, begin = 0, end = 1, option = "plasma"),
                 # cellnote = TRUE,
                 draw_cellnote = TRUE,
                 limits = c(0.0, 0.08),
                 digits = 4L,
                 # colors = c("blue", "white", "red"),
                 cellnote_textposition = "middle center",
                 # scale_fill_gradient_fun = ggplot2::scale_fill_gradient2(low = "red",
                 #                                                         high = "blue",
                 #                                                         mid = "white",
                 #                                                         midpoint = 1.0,
                 #                                                         limits = c(1.0, 5.0)),
                 # scale_fill_gradient_fun = ggplot2::scale_fill_gradientn(colors = c("red", "white", "blue"),
                 #                                                         values = rescale(c(0, 1, 4.7)),
                 #                                                         limits=c(0, 4.7)),
                 scale = "none",
                 na.value = "grey50",
                 # limits = c(0.5, 1.0),
                 # margins = c(60,100,NA,NA),
                 grid_color = "white",
                 grid_width = 0.00002,
                 titleX = TRUE,
                 hide_colorbar = FALSE,
                 branches_lwd = NULL,
                 label_names = c("Development", "Validation:", "AUROC"),
                 fontsize_row = 10,
                 fontsize_col = 10,
                 labCol = colnames(data),
                 labRow = rownames(data),
                 heatmap_layers = theme(axis.line=element_blank()))
  print(p)
}

drawObsTarHeatmap <- function(data, title = "Heatmap") {
  
  rownames(data) <- c("365 days", "730 days", "1095 days")
  colnames(data) <- c("365 days", "1095 days", "1825 days")
  
  p <- heatmaply(data,
                 dendrogram = "none",
                 ylab = "Observation period",
                 xlab = "Time at risk", 
                 main = title,
                 # cellnote = TRUE,
                 draw_cellnote = TRUE,
                 # limits = c(0.5, 1.0),
                 digits = 2L,
                 cellnote_textposition = "middle center",
                 scale = "none",
                 na.value = "grey50",
                 limits = c(0.5, 1.0),
                 # margins = c(60,100,NA,NA),
                 grid_color = "white",
                 grid_width = 0.00002,
                 titleX = TRUE,
                 hide_colorbar = FALSE,
                 branches_lwd = NULL,
                 label_names = c("Development", "Validation:", "AUROC"),
                 fontsize_row = 10,
                 fontsize_col = 10,
                 labCol = colnames(data),
                 labRow = rownames(data),
                 heatmap_layers = theme(axis.line=element_blank()))
  print(p)
}

drawReplicationHeatmap <- function(data, title = "Heatmap") {
  
  rownames(data) <- c("THIN (Walters)")
  colnames(data) <- c("MDCR", "IqviaGer", "OptumSES", "OptumPanther", "CPRD", "IPCI", "THIN (Walters)")
  
  p <- heatmaply(data,
                 dendrogram = "none",
                 ylab = "Development database",
                 xlab = "Validation database", 
                 main = title,
                 # cellnote = TRUE,
                 draw_cellnote = TRUE,
                 # limits = c(0.5, 1.0),
                 digits = 2L,
                 cellnote_textposition = "middle center",
                 scale = "none",
                 na.value = "grey50",
                 limits = c(0.5, 1.0),
                 # margins = c(60,100,NA,NA),
                 grid_color = "white",
                 grid_width = 0.00002,
                 titleX = TRUE,
                 hide_colorbar = FALSE,
                 branches_lwd = NULL,
                 label_names = c("Development", "Validation:", "AUROC"),
                 fontsize_row = 10,
                 fontsize_col = 10,
                 labCol = colnames(data),
                 labRow = rownames(data),
                 heatmap_layers = theme(axis.line=element_blank()),
                 height = 100,
                 width = 400)
  print(p)
}

drawEavgHeatmap <- function(data, title = "Heatmap") {
  
  rownames(data) <- c("MDCR", "IQGER", "OPSES", "OPEHR", "IPCI")
  colnames(data) <- c("MDCR", "IQGER", "OPSES", "OPEHR", "IPCI")
  
  p <- heatmaply(data,
                 dendrogram = "none",
                 ylab = "Development database",
                 xlab = "Validation database", 
                 main = title,
                 # cellnote = TRUE,
                 draw_cellnote = TRUE,
                 # limits = c(0.5, 1.0),
                 digits = 2L,
                 cellnote_textposition = "middle center",
                 scale = "none",
                 na.value = "grey50",
                 limits = c(0.0, 0.15),
                 # margins = c(60,100,NA,NA),
                 grid_color = "white",
                 grid_width = 0.00002,
                 titleX = TRUE,
                 hide_colorbar = FALSE,
                 branches_lwd = NULL,
                 label_names = c("Development", "Validation:", "AUROC"),
                 fontsize_row = 10,
                 fontsize_col = 10,
                 labCol = colnames(data),
                 labRow = rownames(data),
                 heatmap_layers = theme(axis.line=element_blank()))
  print(p)
}
