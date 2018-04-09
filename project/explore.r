# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., title=NULL, plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]] + ggtitle(NULL), vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


replot = TRUE


data = read.csv('rbench.csv')
data = data[which(!grepl("harness", data$benchmark)),]

library(e1071)
#library(kernlab)
library(ggplot2)

bm_names = sort(unique(data$benchmark))

l = length(bm_names)
w = length(data[data$benchmark == bm_names[[1]],1])
w = w - w%%100

clustering <- matrix(vector("numeric", l*w), ncol = w)

for (bm in bm_names) {
  cat(bm, "\n")

  s = data[data$benchmark == bm,]

  # remove half done runs
  s = s[1:(length(s$time)-(length(s$time)%%100)),]

  #remove unnecessary rows
  s = s[,(names(s) %in% c("time","R_NGrowIncrFrac","R_NGrowFrac","R_MaxKeepFrac","R_MinFreeFrac"))]
  ii = which(bm_names == bm)

  s$good_time = TRUE
#  s$good_mem = 2
  s$good_time[which(s$time > min(s$time)*1.03)] = FALSE
#  s$good_mem[which(s$mem > min(s$mem)*1.04)] = 1

  s$good_time = factor(s$good_time)

  if(replot) {
      plots2 <- list(
        ggplot(s, aes(x=R_NGrowFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=time), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(s, aes(x=R_NGrowIncrFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=time), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(s, aes(x=R_MaxKeepFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=time), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),

        ggplot(s, aes(x=R_NGrowFrac, y=R_NGrowIncrFrac)) +
                     geom_tile(aes(fill=time), alpha=0.075) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(s, aes(x=R_MaxKeepFrac, y=R_NGrowIncrFrac)) +
                     geom_tile(aes(fill=time), alpha=0.075) +
                      ggtitle(bm) + theme(legend.position="none"),

        ggplot(s, aes(x=R_NGrowFrac, y=R_MaxKeepFrac)) +
                     geom_tile(aes(fill=time), alpha=0.075) +
                      ggtitle(bm) + theme(legend.position="none"))

       jpeg(filename = paste("out/",bm,"_real.jpg",sep=""), width=1024, height=1024)
       multiplot(title=bm, plotlist=plots2, layout=matrix(c(2,3,1,0,5,4,0,0,6),nrow=3))
       dev.off()

      plots2 <- list(
        ggplot(s, aes(x=R_NGrowFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=good_time), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(s, aes(x=R_NGrowIncrFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=good_time), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(s, aes(x=R_MaxKeepFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=good_time), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),

        ggplot(s, aes(x=R_NGrowFrac, y=R_NGrowIncrFrac)) +
                     geom_tile(aes(fill=good_time), alpha=0.075) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(s, aes(x=R_MaxKeepFrac, y=R_NGrowIncrFrac)) +
                     geom_tile(aes(fill=good_time), alpha=0.075) +
                      ggtitle(bm) + theme(legend.position="none"),

        ggplot(s, aes(x=R_NGrowFrac, y=R_MaxKeepFrac)) +
                     geom_tile(aes(fill=good_time), alpha=0.075) +
                      ggtitle(bm) + theme(legend.position="none"))

       jpeg(filename = paste("out/",bm,"_class.jpg",sep=""), width=1024, height=1024)
       multiplot(title=bm, plotlist=plots2, layout=matrix(c(2,3,1,0,5,4,0,0,6),nrow=3))
       dev.off()

  }
  s = s[, !(names(s) %in% c("time"))]


  ratio = length(which(s$good_time == TRUE)) / length(s$good_time)
  if (ratio > 0.95 || ratio < 0.05) {
    cat(bm, " has no time dependencies\n")
    clustering[ii,1] <- -10000
    clustering[ii,2] <- 10000
  } else {

    tuned = tune.svm(good_time~., data =s, gamma=seq(0.02,0.12,0.01), cost=seq(0.5,1.5,0.3),
                     tunecontrol=tune.control(cross=10),
                     type="C-classification", kernel="radial")
    #m = svm(good_time~., data =s, type="C-classification", kernel="radial")
    print(tuned)

    m = tuned$best.model
    #m = svm(good_time ~ R_NGrowIncrFrac + R_NGrowFrac + R_MaxKeepFrac + R_MinFreeFrac, data=s,type="C-classification", kernel="radial", cost=0.5)
    # m = svm(good_time ~., data=s,type="C-classification", kernel="radial", gamma=0.02, cost=0.5)

    if (length(clustering[ii,]) > length(m$decision.values))
      clustering[ii,] <- c(m$decision.values, m$decision.values[[length(m$decision.values)]])
    else
      clustering[ii,] <- m$decision.values

    if (replot) {
      f <- expand.grid(
        R_NGrowIncrFrac = seq(min(s$R_NGrowIncrFrac)-0.1, max(s$R_NGrowIncrFrac)+0.1, 0.03),
        R_NGrowFrac = seq(min(s$R_NGrowFrac)-0.1, max(s$R_NGrowFrac)+0.1, 0.03),
        R_MaxKeepFrac = seq(min(s$R_MaxKeepFrac)-0.1, max(s$R_MaxKeepFrac)+0.1, 0.03),
        R_MinFreeFrac = seq(min(s$R_MinFreeFrac)-0.1, max(s$R_MinFreeFrac)+0.1, 0.03)
      )

      f$prediction = predict(m, newdata=f)

      plots1 <- list(
        ggplot(f, aes(x=R_NGrowFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=prediction), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(f, aes(x=R_NGrowIncrFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=prediction), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(f, aes(x=R_MaxKeepFrac, y=R_MinFreeFrac)) +
                     geom_tile(aes(fill=prediction), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(f, aes(x=R_NGrowFrac, y=R_NGrowIncrFrac)) +
                     geom_tile(aes(fill=prediction), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),
        ggplot(f, aes(x=R_MaxKeepFrac, y=R_NGrowIncrFrac)) +
                     geom_tile(aes(fill=prediction), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"),

        ggplot(f, aes(x=R_NGrowFrac, y=R_MaxKeepFrac)) +
                     geom_tile(aes(fill=prediction), alpha=0.05) +
                      ggtitle(bm) + theme(legend.position="none"));

   jpeg(filename = paste("out/",bm,"_svm.jpg",sep=""), width=1024, height=1024)
   multiplot(title=bm, plotlist=plots1, layout=matrix(c(2,3,1,0,5,4,0,0,6),nrow=3))
   dev.off()
    }
 }
}

library(NbClust)
ks = kmeans(clustering, 16, nstart=8000)

cat(paste(ks$cluster, collapse="\n"),"\n")


for (i in 1:16) {
  cat("==========\n");
  for (ii in 1:length(bm_names)) {
    if (ks$cluster[[ii]] == i)
      cat(" * ", as.character(bm_names[[ii]]), "\n")
  }
}


print(ks$betweenss/ ks$totss)
