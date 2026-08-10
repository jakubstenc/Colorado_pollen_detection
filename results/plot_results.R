library(ggplot2)
library(dplyr)
library(readr)

# 1. Plot Pollen Deposition
if (file.exists("pollen_deposition_aggregated.csv")) {
  dep_data <- read_csv("pollen_deposition_aggregated.csv", show_col_types = FALSE)
  
  # Convert to proper Date format
  dep_data$Collection_Date <- as.Date(dep_data$Collection_Date)
  
  # Reshape data to plot Conspecific and Heterospecific easily if needed, 
  # or plot them as separate lines. Here we do it manually or via reshaping.
  library(tidyr)
  dep_long <- dep_data %>% 
    select(Collection_Date, Species, Conspecific_Avg, Conspecific_SE, Heterospecific_Avg, Heterospecific_SE) %>%
    pivot_longer(
      cols = c(Conspecific_Avg, Heterospecific_Avg, Conspecific_SE, Heterospecific_SE),
      names_to = c("Type", ".value"),
      names_pattern = "(.*)_(.*)"
    )
  
  p1 <- ggplot(dep_long, aes(x = Collection_Date, y = Avg, color = Species, linetype = Type, shape = Type)) +
    geom_point(size = 3) +
    geom_line(linewidth = 1) +
    geom_errorbar(aes(ymin = Avg - SE, ymax = Avg + SE), width = 0.2) +
    labs(
      title = "Pollen Deposition Over Time (Conspecific vs Heterospecific)",
      x = "Collection Date",
      y = "Average Grains per Image"
    ) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "right",
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("pollen_deposition_timeline_R.png", p1, width = 10, height = 6, dpi = 300)
  print("Saved pollen_deposition_timeline_R.png")
}

# 2. Plot Pollen Production
if (file.exists("pollen_production_aggregated.csv")) {
  prod_data <- read_csv("pollen_production_aggregated.csv", show_col_types = FALSE)
  
  prod_data$Collection_Date <- as.Date(prod_data$Collection_Date)
  
  p2 <- ggplot(prod_data, aes(x = Collection_Date, y = Average_Detections, color = Species)) +
    geom_point(size = 3) +
    geom_line(linewidth = 1) +
    geom_errorbar(aes(ymin = Average_Detections - Standard_Error, ymax = Average_Detections + Standard_Error), width = 0.2) +
    labs(
      title = "Pollen Production Over Time",
      x = "Collection Date",
      y = "Average Grains per Anther"
    ) +
    scale_y_log10() +
    theme_minimal(base_size = 14) +
    theme(legend.position = "right",
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("pollen_production_timeline_R.png", p2, width = 10, height = 6, dpi = 300)
  print("Saved pollen_production_timeline_R.png")
}
