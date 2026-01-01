library(afex)
library(ggplot2)
library(emmeans)
library(ggstatsplot)

options(width=100)

dataset <- read.csv("results_grid_T13StereoFix.csv", stringsAsFactors = TRUE)
# dataset <- dataset[dataset$Device == "Ring" & dataset$Cams != 1 & dataset$Metric == "Traj_RMSE",]
# # dataset <- dataset[dataset$Device == "Ring" & dataset$Cams != 1 & dataset$Metric == "Traj_RMSE",]
dataset <- dataset[dataset$Device == "Ring",]
summary(dataset)

# ggbetweenstats(data=dataset, x=Cams, y=Error, type="nonparametric", pairwise.display = "all")
plot <- grouped_ggbetweenstats(data=dataset, x=Cams, y=Error, grouping.var = Metric, type="nonparametric", pairwise.display = "all", plotgrid.args = list(ncol = 3, axes = "collect_y"), var.equal = FALSE, ggplot.component = ylim(0, 20))
ggsave(plot = plot, width = 18, height = 12, dpi = 300, filename = "Cams, outliers removed.pdf")

# dataset$Cams <- as.factor(dataset$Cams)

# contrasts(dataset$Tracking) <- c(1/2, -1/2)

# result <- lmer(Error ~ (Cams + Tracking + Fusion)^2 + (1 | Config) + 0, data = dataset) # nolint: line_length_linter.
# result <- lm(Error ~ (Cams + Tracking + Fusion)^2 + 0, data = dataset)
# # result <- glm(Error ~ (Cams + Tracking + Fusion)^2 + 0, data = dataset, family="Gamma")
# summary(result)

# dataset$pred <- predict(result)
# ggplot(dataset,aes(x=Cams,y=pred,colour=Config, group=Config)) + geom_point() + geom_line() + theme(legend.position="bottom", legend.direction = "horizontal")

# em_fusion = emmeans(result, ~ Fusion, lmerTest.limit = 49920)
# em_fusion
# co_fusion = contrast(em_fusion, "consec", simple = "each", combine = FALSE, adjust = "mvt") # nolint: line_length_linter.
# co_fusion

# em_tracking = emmeans(result, ~ Tracking, lmerTest.limit = 49920)
# em_tracking
# co_tracking = contrast(em_tracking, "consec", simple = "each", combine = FALSE, adjust = "mvt")
# co_tracking

# em_cams = emmeans(result, ~ Cams, lmerTest.limit = 49920)
# em_cams
# co_cams = contrast(em_cams, "consec", simple = "each", combine = FALSE, adjust = "mvt")
# co_cams

# em_all = emmeans(result, ~ Cams * Tracking * Fusion, lmerTest.limit = 49920)
# em_all
# co_all = contrast(em_all, "consec", simple = "each", combine = FALSE, adjust = "tukey")
# co_all

# eff = eff_size(em_all, method="pairwise", sigma=sigma(result), edf=49902)