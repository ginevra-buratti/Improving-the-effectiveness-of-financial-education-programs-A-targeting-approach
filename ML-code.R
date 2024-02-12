library(tidyverse)

library(rpart.plot) # for the decision tree
library(randomForest) # for the random forest
library(ROCR) # ROC, LIFT



# Data preparation --------------------------------------------------------


# read data
data <- read.csv("ML_ex_ante.csv")

# remove alternative definitions of the outcome variable and not employed predictors
data <- data %>% select(-c(y, y_v2, y_upas, y_upas_v2, y_upas_v4))
data <- data %>% select(-c(q2))
data <- data %>% select(-c(q4))
data <- data %>% select(-c(q5_1_1, q5_1_2, q5_1_3, q5_1_4, q5_1_5, q5_1_6))
data <- data %>% select(-c(q5_2, q5_3))
data <- data %>% select(-c(q7))
data <- data %>% select(-c(q25_1))
data <- data %>% select(-c(q42_1, q42_2, q42_3, q42_4, q42_5, q42_6))

# rename variables
colnames(data)[5] <- "female"
colnames(data)[6] <- "age"
colnames(data)[23] <- "q5_max"

# variables as factor
data$y_upas_v3 <- as.factor(data$y_upas_v3)
data$female <- as.factor(data$female)
data$age <- as.factor(data$age)
data$q2_1 <- as.factor(data$q2_1)
data$q2_2 <- as.factor(data$q2_2)
data$q2_3 <- as.factor(data$q2_3)
data$q2_4 <- as.factor(data$q2_4)
data$q2_5 <- as.factor(data$q2_5)
data$q2_6 <- as.factor(data$q2_6)
data$q3_1 <- as.factor(data$q3_1)
data$q3_2 <- as.factor(data$q3_2)
data$q4_1 <- as.factor(data$q4_1)
data$q4_2 <- as.factor(data$q4_2)
data$q4_3 <- as.factor(data$q4_3)
data$q4_4 <- as.factor(data$q4_4)
data$q4_5 <- as.factor(data$q4_5)
data$q4_6 <- as.factor(data$q4_6)
data$q4_7 <- as.factor(data$q4_7)
data$q5_1 <- as.factor(data$q5_1)
data$q5_max <- as.factor(data$q5_max)
data$q6 <- as.factor(data$q6)
data$q7_1 <- as.factor(data$q7_1)
data$q7_2 <- as.factor(data$q7_2)
data$q7_3 <- as.factor(data$q7_3)
data$q7_4 <- as.factor(data$q7_4)
data$q24_1 <- as.factor(data$q24_1)
data$q24_other <- as.factor(data$q24_other)
data$q24_10 <- as.factor(data$q24_10)

table(data$group)

# split data into training (ind=1) & holdout (ind=2) sample
# we require the holdout sample to have an equal number of untreated and T1-treated individuals

# all untreated individuals (group=0) who claim to usually watch UPAS (q50_2=1) are assigned to the training sample, the remaining individuals are randomly assigned to either of the two samples
set.seed(314159)
ind0 <- ifelse(data[data$group==0,]$q50_2==0, sample(2, nrow(data[data$group==0,]), replace=TRUE, prob=c(0.4006, 0.5994)), 1)
table(ind0)

# all T1-treated individuals (group=1) who claim to usually watch UPAS (q50_2=1) are assigned to the holdout sample, the remaining individuals are randomly assigned to either of the two samples
set.seed(314159)
ind1 <- ifelse(data[data$group==1,]$q50_2==1, 2, sample(2, nrow(data[data$group==1,]), replace=TRUE, prob=c(0.544, 0.456)))
table(ind1)

# all T2- and T3-treated individuals are assigned to the training sample
ind2 <- rep(1, times=nrow(data[data$group==2,]))
ind3 <- rep(1, times=nrow(data[data$group==3,]))

ind <- c(ind0, ind1, ind2, ind3)

firstcol <- 5
x <- as.data.frame(data[firstcol:ncol(data)])

x_train <- x[ind==1,]
x_test <- x[ind==2,]

y_train <- data[ind==1, colnames(data)=="y_upas_v3"]
y_test <- data[ind==2, colnames(data)=="y_upas_v3"]

table(y_train)
table(y_test)

# variables as ordered

x_train$age <- ordered(x_train$age, levels=c("1", "2", "3"))
x_train$q3_1 <- ordered(x_train$q3_1, levels=c("0", "1", "2", "3", "4", "5", "6"))
x_train$q3_2 <- ordered(x_train$q3_2, levels=c("0", "1", "2", "3", "4", "5", "6", "7"))
x_train$q5_1 <- ordered(x_train$q5_1, levels=c("1", "2", "3", "4", "5", "6"))
x_train$q5_max <- ordered(x_train$q5_max, levels=c("1", "2", "3", "4", "5", "6"))
x_train$q6 <- ordered(x_train$q6, levels=c("1", "2", "3", "4", "5", "6", "7"))

x_test$age <- ordered(x_test$age, levels=c("1", "2", "3"))
x_test$q3_1 <- ordered(x_test$q3_1, levels=c("0", "1", "2", "3", "4", "5", "6"))
x_test$q3_2 <- ordered(x_test$q3_2, levels=c("0", "1", "2", "3", "4", "5", "6", "7"))
x_test$q5_1 <- ordered(x_test$q5_1, levels=c("1", "2", "3", "4", "5", "6"))
x_test$q5_max <- ordered(x_test$q5_max, levels=c("1", "2", "3", "4", "5", "6"))
x_test$q6 <- ordered(x_test$q6, levels=c("1", "2", "3", "4", "5", "6", "7"))



# Classification tree -----------------------------------------------------


set.seed(314159) 
tree <- rpart(y_train ~., method="class", data= x_train, cp=0.001)

# plot cp
plotcp(tree)

# print cp table 
printcp(tree)

# select minimum xerror
cp_table <- as.data.frame(tree$cptable)
xerror_min <- cp_table[which.min(cp_table[,"xerror"]),"xerror"]

# use Hastie rule to prune the tree
xstd_min <- cp_table[which.min(cp_table[,"xerror"]),"xstd"]
cp_min <- cp_table[which.min(cp_table[,"xerror"]),"CP"]
cp_list <- cp_table[cp_table$xerror>xerror_min+xstd_min & cp_table$CP>cp_min,]$CP
cp_tree <- cp_list[which.min(cp_list)]
pruned_tree <- prune(tree, cp_tree)
pdf("tree.pdf", width=4, height=3)
prp(pruned_tree,fallen.leaves = FALSE, box.col="lightgray", type = 3, branch=1, branch.lty= 1,   main="Classification Tree for Needy Learners", digits=5)
dev.off()

# write to output file
sink("tree.txt")
str(x_train)
printcp(tree)
cat(paste("\n",sep="\n"))
y_pred <- predict(pruned_tree, x_test, type="class")
assess <- data.frame(y = y_test, y_pred = y_pred)
assess <- assess %>% mutate(correct = y == y_pred)
assess %>% summarize(n = n(), per = sum(correct)/n())
cat(paste("\n",sep="\n"))
assess %>% group_by(y) %>% summarize(n = n(), per = sum(correct)/n())
sink()

# variable importance plot

var_impo_tree <- pruned_tree$variable.importance %>% 
  data.frame() %>%
  rownames_to_column(var = "var_name")

var_impo_tree <- rename(var_impo_tree, var_impo = .)
var_impo_tree$rel_impo = var_impo_tree$var_impo/var_impo_tree$var_impo[1]*100
most_impo_tree <- var_impo_tree[1:10,]

pdf("varimpo_tree.pdf")
ggplot(most_impo_tree, aes(x = fct_reorder(var_name, rel_impo), y = rel_impo)) +
  geom_point(shape = 21, color = "black", fill = "white") +
  theme_bw() +
  coord_flip() +
  labs(x = "", y = "") +
  theme(panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey",
                                          size = 0.2,
                                          linetype = 3),
        axis.ticks.y = element_blank(),
        aspect.ratio=5/4)
dev.off()



# Random forest -----------------------------------------------------------


set.seed(314159)
rf <- randomForest(y_train ~., data= x_train, importance=TRUE, ntree=1000)

# variable importance plot

importance(rf)

pdf("varimpo_rf.pdf")
varImpPlot(rf, sort=TRUE, n.var=min(10, nrow(rf$importance)), main="")
dev.off()



# Export predictions ------------------------------------------------------


# variables as ordered
x$age <- ordered(x$age, levels=c("1", "2", "3"))
x$q3_1 <- ordered(x$q3_1, levels=c("0", "1", "2", "3", "4", "5", "6"))
x$q3_2 <- ordered(x$q3_2, levels=c("0", "1", "2", "3", "4", "5", "6", "7"))
x$q5_1 <- ordered(x$q5_1, levels=c("1", "2", "3", "4", "5", "6"))
x$q5_max <- ordered(x$q5_max, levels=c("1", "2", "3", "4", "5", "6"))
x$q6 <- ordered(x$q6, levels=c("1", "2", "3", "4", "5", "6", "7"))

predictions_tree <- predict(pruned_tree, x, type="class")
predictions_rf <- predict(rf, x)
output <- data.frame(ind = ind, y_tree = predictions_tree, y_rf = predictions_rf)

write.csv(output,"ML_pred.csv")



# Model performance -------------------------------------------------------


# roc curves

pdf("roc.pdf", width=5, height=5)

pred_tree <- prediction(predict(pruned_tree, x_test, type = "prob")[, 2], y_test)
plot(performance(pred_tree, "tpr", "fpr"), lty = 2)

pred_rf <- prediction(predict(rf, x_test, type = "prob")[, 2], y_test)
plot(performance(pred_rf, "tpr", "fpr"), add = TRUE)

pred_lpm <- prediction(read.csv("lpm.csv")[, 2], read.csv("lpm.csv")[, 1])
plot(performance(pred_lpm, "tpr", "fpr"), lty = 4, add = TRUE)

abline(0, 1, lty = 3, col = "snow4")

legend(x = "bottomright",                                    # Position
       legend = c("Decision tree", "Random forest", "LPM"),  # Legend texts
       lty = c(2, 1, 4))                                     # Line types

dev.off()

# AUC

auc_tree <- performance(pred_tree, measure = "auc")
auc_tree <- auc_tree@y.values[[1]]
print(auc_tree)

auc_rf <- performance(pred_rf, measure = "auc")
auc_rf <- auc_rf@y.values[[1]]
print(auc_rf)

auc_lpm <- performance(pred_lpm, measure = "auc")
auc_lpm <- auc_lpm@y.values[[1]]
print(auc_lpm)



# Robustness exercise -----------------------------------------------------


# read data
data <- read.csv("ML_robustness.csv")

# variables as factors/ordered factors
data$y_upas_v3 <- as.factor(data$y_upas_v3)
data$sesso <- as.factor(data$sesso)
data$eta <- as.factor(data$eta)
data$eta <- ordered(data$eta, levels = c("1", "2", "3"))
data$q2_1 <- as.factor(data$q2_1)
data$q2_2 <- as.factor(data$q2_2)
data$q2_3 <- as.factor(data$q2_3)
data$q2_4 <- as.factor(data$q2_4)
data$q2_5 <- as.factor(data$q2_5)
data$q2_6 <- as.factor(data$q2_6)
data$q3_1 <- as.factor(data$q3_1)
data$q3_1 <- ordered(data$q3_1, levels = c("0", "1", "2", "3", "4", "5", "6"))
data$q3_2 <- as.factor(data$q3_2)
data$q3_2 <- ordered(data$q3_2, levels = c("0", "1", "2", "3", "4", "5", "6", "7"))
data$q4_1 <- as.factor(data$q4_1)
data$q4_2 <- as.factor(data$q4_2)
data$q4_3 <- as.factor(data$q4_3)
data$q4_4 <- as.factor(data$q4_4)
data$q4_5 <- as.factor(data$q4_5)
data$q4_6 <- as.factor(data$q4_6)
data$q4_7 <- as.factor(data$q4_7)
data$q5_1 <- as.factor(data$q5_1)
data$q5_1 <- ordered(data$q5_1, levels = c("1", "2", "3", "4", "5", "6"))
data$q5_gen <- as.factor(data$q5_gen)
data$q5_gen <- ordered(data$q5_gen, levels = c("1", "2", "3", "4", "5", "6"))
data$q6 <- as.factor(data$q6)
data$q6 <- ordered(data$q6, levels = c("1", "2", "3", "4", "5", "6", "7"))
data$q7_1 <- as.factor(data$q7_1)
data$q7_2 <- as.factor(data$q7_2)
data$q7_3 <- as.factor(data$q7_3)
data$q7_4 <- as.factor(data$q7_4)
data$q24_1 <- as.factor(data$q24_1)
data$q24_other <- as.factor(data$q24_other)
data$q24_10 <- as.factor(data$q24_10)

# count individuals in each group
table(data$group)

# count control and T1-treated individuals
table(data[data$group == 0, ]$q50_2)
table(data[data$group == 1, ]$q50_2)

n_it <- 100

ind0 = matrix(, nrow = 965, ncol = n_it)
ind1 = matrix(, nrow = 960, ncol = n_it)
ind = matrix(, nrow = 3855, ncol = n_it)

ind2 <- rep(1, times=nrow(data[data$group == 2, ]))
ind3 <- rep(1, times=nrow(data[data$group == 3, ]))

firstcol <- 8
x <- as.data.frame(data[firstcol:ncol(data)])

output <- data.frame(id = data$id)

set.seed(314159)

for (i in 1:n_it) {
  print(i)
  
  # split data into training (ind=1) and holdout (ind=2) sample
  ind0[, i] <- ifelse(data[data$group == 0, ]$q50_2 == 0, sample(2, nrow(data[data$group == 0, ]), replace = TRUE, prob = c(0.4104, 0.5896)), 1)
  ind1[, i] <- ifelse(data[data$group == 1, ]$q50_2 == 1, 2, sample(2, nrow(data[data$group == 1, ]), replace = TRUE, prob = c(0.5418, 0.4582)))
  ind[, i] <- c(ind0[, i], ind1[, i], ind2, ind3)
  
  # random forest
  
  x_train <- x[ind[, i] == 1, ]
  x_test <- x[ind[, i] == 2, ]
  
  y_train <- data[ind[, i] == 1, colnames(data) == "y_upas_v3"]
  y_test <- data[ind[, i] == 2, colnames(data) == "y_upas_v3"]
  
  rf <- randomForest(y_train ~ ., data = x_train, importance = TRUE, ntree = 1000)
  predictions_rf <- predict(rf, x)
  
  output$ind <- ind[, i]
  output$target <- predictions_rf
  
  # export training/holdout labels and ML predictions
  filename <- paste(paste("ML_draw_ ", i, sep = ""), "csv", sep = ".")
  write.csv(output, filename, row.names = FALSE)
  
}