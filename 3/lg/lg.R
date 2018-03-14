library(R.matlab)
library(ggplot2)

lg <- function(name, store) {
  data <- readMat(name)
  train = data.frame(y=factor(data$Y.trn), x=data$X.trn)
  test = data.frame(y=factor(data$Y.tst), x=data$X.tst)
  model <- glm (y ~ 0+x.1+x.2, data=train, family = binomial)
  print(summary(model))

  predict_train <- predict(model, type = 'response')
  predict_test <- predict(model, newdata=test, type = 'response')

  table(train$y, predict_train > 0.5)
  table(test$y, predict_test > 0.5)

  # for non zero intercept
  # f <- function(x) (-model$coefficients[[1]]/model$coefficients[[3]])+
  #                   x*(-model$coefficients[[2]]/model$coefficients[[3]])
  f <- function(x) x*(-model$coefficients[[1]]/model$coefficients[[2]])

  res <- data.frame(
      x.1=c(train$x.1, test$x.1),
      x.2=c(train$x.2, test$x.2),
      y=c(
          sapply(train$y, function(x) if(x==0) "train1" else "train0"),
          sapply(test$y, function(x) if(x==0) "test1" else "test0"))
  )

  ggplot(res, aes(x=x.1, y=x.2, color=y)) + geom_point() +
    stat_function(fun=f, n=100000, color="black") + ylim(min(res$x.2), max(res$x.2)) +
  ggsave(store)
}


lg('data/data1.mat', "data1_r.png")
lg('data/data2.mat', "data2_r.png")
