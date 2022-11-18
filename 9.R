library(MLmetrics)
library(Metrics)

actual <- c(12, 13, 14, 15, 15, 22, 27)
predicted <- c(11, 13, 14, 14, 15, 16, 18)
smape(actual,predicted)

dat <- data.frame(
  actual = c(12, 13, 14, 15, 15, 22, 27),
  predicted = c(11, 13, 14, 14, 15, 16, 18)
)

my.function <- function(data, predict, actual) {
  a = MAPE(data$predict, data$actual)
  b = mae(data$actual,data$predict)
  c = rmse(data$actual, data$predicted)
  d = smape(data$actual, data$predict)
  return(c(a, b, c, d))
}

my.function(dat, "predicted", "actual")

