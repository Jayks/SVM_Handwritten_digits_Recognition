##***************************************************************************##
####            Support Vector Machines - Pattern Recognition              ####
##***************************************************************************##
## Objective      : To develop a model using Support Vector Machines which   ## 
##                  should correctly classify the handwritten digits based   ##
##                  on the pixel values given as features.                   ##
## Date           : 24-Jun-2018                                              ##
## Version        : 1.0                                                      ##
##***************************************************************************##

######################   Business understanding  ##############################
# A classic problem in the field of pattern recognition is that of 
# handwritten digit recognition. The goal here is to develop a model that can 
# correctly identify the digit (between 0-9) written in an image which is 
# submitted by a user via a scanner, a tablet, or other digital devices. 


##################  Installing & Loading required packages ####################
# Check and Import required libraries
options(warn = -1)
libs = c("tidyverse", "formattable", "caret", "kernlab",
         "gridExtra", "splitstackshape", "cowplot")
install.lib <- libs[!libs %in% installed.packages()]
for (pkg in install.lib)
  install.packages(pkg, dependencies = T)
loadlib     <- lapply(libs, library, character.only = T) # load them
remove(list = ls())
options(warn = 0)


############################  Import Datasets  ################################
# Loading files and adding column names
train <- read_csv("mnist_train.csv", 
                  col_names = c("label", paste0("pix", 1:784)))
test  <- read_csv("mnist_test.csv", 
                  col_names = c("label", paste0("pix", 1:784)))


############################  Data Understanding ##############################
dim(train) 
# 60,000 observations with 785 columns in train dataset. 
dim(test)
# 10,000 observations with 785 columns. 
# Label columns is the target variableto be predicted 
head(train[ ,1:10])
head(test[ ,1:10])
tail(train[,1:10])
tail(test[,1:10])

# Checking for NA values 
anyNA(train)
anyNA(test)
# No NA Values

# Checking for Duplicate values 
sum(duplicated(rbind(train, test)))
# 0 - Implies no duplicates 

# Lets visualize each digits taking some random sample observations
for (i in seq(0, 9)) {
  sample <- train[train$label == i, ]
  # Omit label column 
  sample <-  sample[ ,-1]
  # Resetting the margins
  par(mar = c(1,1,1,1))  
  # Build 10 rows by 5 column Plot matrix  for each digit 
  par(mfrow = c(5,10)) 
  # 50 samples between 10 & 5000 
  for (j in seq(10, 5000, by = 100)) {
    # Build a 28 X 28 matrix of pixel values in each row
    digit <- t(matrix(as.numeric(sample[j, ]), nrow = 28)) 
    # Inverse the pixel matrix to get the image of the number right
    image(t(apply(digit, 2, rev)), col = grey.colors(255))
  }
}
# Deleting temporary varaibles 'digit', sample' , i , j 
remove(digit, sample, i ,j)


########################  Derive New variables   ##############################
# Lets derive the average intensity of each number
train$AvgIntensity <-  rowMeans(train[,-1])
test$AvgIntensity <-  rowMeans(test[,-1])

# Convert the target variables to factors
train$label <- factor(train$label)
test$label <- factor(test$label)


###########################        EDA           ##############################
# Setting theme 
theme_set(theme_classic() + 
          theme(plot.title = element_text(hjust = 0.5, size = 12,face = 'bold'),
                axis.title.x = element_text(size = 12),
                axis.title.y = element_text(size = 12),
                axis.text.x  = element_text(size = 10),
                axis.text.y  = element_text(size = 10),
                legend.position = 'none'))

plot_grid(
  train %>% 
    group_by(label) %>% 
    summarize(AvgInt = mean(AvgIntensity)) %>% 
    ggplot(aes(x = factor(label), y = AvgInt, fill = factor(label))) + 
    geom_col() + 
    labs(x = "Numbers", title = "Number vs Avg. Intensity in Train Dataset"), 
  test %>% 
    group_by(label) %>% 
    summarize(AvgInt = mean(AvgIntensity)) %>% 
    ggplot(aes(x = factor(label), y = AvgInt, fill = factor(label))) + 
    geom_col() + 
    labs(x = "Numbers", title = "Number vs Avg. Intensity in Test Dataset"), 
    nrow = 2)
# As we can see there are some differences in intensity. 
# The digit 1 is the less intense while the digit 0 is the most intense. 
# The AvgIntensity can be used as one of the predictor variable
# Also both test and train datasets are stratified samples from the
# orginal population.

plot_grid(
  train %>% 
    ggplot(aes(x = factor(label), y = AvgIntensity, fill = factor(label))) + 
    geom_violin() + 
    labs(x = "Numbers", title = "Avg. Intensity Distribution in Train Dataset"), 
  test %>% 
    ggplot(aes(x = factor(label), y = AvgIntensity, fill = factor(label))) + 
    geom_violin() + 
    labs(x = "Numbers", title = "Avg. Intensity Distribution in Test Dataset"), 
    nrow = 2)
# Most intensity distributions seem roughly normally distributed but some have 
# higher variance than others. The digit 1 seems to be the one people 
# write most consistently followed by 9, 7 & 4. 
# Also both test and train datasets maintain the same distributions


############################  Data Preparation   ##############################
# Prepare Datasets for Model building 

# Commenting below code since removing columns with all constant values 
# did not make any difference with respect to the model metrics

# #######Start comment 
# # As we saw during visualization there might be possibilities of 
# # a column having all 0s. Lets check if there are any constant 0 values 
# # across columns in both train and test datasets
# (zero_value_cols  <- which(colSums(rbind(train[,-1], test[, -1])) == 0))
# length(zero_value_cols) # 65 
# # Since we have 65 columns across train and test dataset that have only 
# # a constant 0 value lets remove them 
# train = train[ , -(zero_value_cols + 1)]
# test  = test[ , -(zero_value_cols + 1)]# 
# # Lets check the dimensions of the test and train datasets
# dim(train) 
# # 60000 observations and 721 observations
# dim(test)
# # 10000 observations and 721 observations
# ######End comment 

# Since the dataset is huge lets do a stratified sampling that can be used 
# for our modeling 
# Lets check the proportion of the target variables in the dataset
percent(prop.table(table(train$label)))
percent(prop.table(table(test$label)))
# The proportion comes on an average of 10 % 

# Lets do Statified Sampling such that this proprotion is maintained in the 
# sampled dataset
set.seed(100)
sample_train <-  stratified(train, "label", size = 500)

# Lets again check the proportion of the target variables in the Datasets
percent(prop.table(table(sample_train$label)))

# The proportion comes to 10 % across all the labels in train dataset
# matching closely with the proportion of the original dataset. 

# Convert Sample_train to tibble just like other datasets. 
sample_train <- as.tibble(sample_train)

# All the below models were run with and without AvgIntensity column
# and there is no difference in the metrics or final predicted Accuracy
# Hence removing AvgIntensity column from the datasets
sample_train <- sample_train %>% select(-c(AvgIntensity))
train <-  train %>% select(-c(AvgIntensity))
test  <-  test %>% select(-c(AvgIntensity))


##########################   Model Building   #################################

# For all model trainings we will use sample_train dataset
# For all predictions and evaluations we will use complete train & test datasets

#### Linear Kernel ####
(linear_model <- ksvm(label~., data = sample_train, 
                      scaled = F, kernel = "vanilladot"))
#### Linear Model Metrics ####
# parameter : cost C = 1 
# Number of Support Vectors : 1682 

#### RBF Kernel ####
(RBF_model <- ksvm(label~ ., data = sample_train, 
                  scaled = F, kernel = "rbfdot"))
#### RBF Model Metrics ####
# parameter : cost C = 1  
# Hyperparameter : sigma =  1.61909310004045e-07
# Number of Support Vectors : 2408 


##########################   Model Evaluation   ###############################

#### Linear Model Evaluation ####
# Evaluation on complete train dataset
linear_eval_train  <- predict(linear_model, train)
# Confusion matrix for Linear Kernel on original complete train dataset
confusionMatrix(linear_eval_train, train$label)
# Accuracy : 0.9145 

# Evaluation on complete test dataset
linear_eval_test  <- predict(linear_model, test)
# Confusion matrix for Linear Kernel on complete test dataset
confusionMatrix(linear_eval_test, test$label)
# Accuracy : 0.9135 

# The Accuracy is consistent between the whole train and test datasets


#### RBF Model Evaluation ####
# Evaluation on complete train dataset
RBF_eval_train <- predict(RBF_model, train)
# Confusion matrix for RBF Kernel on original complete train dataset
confusionMatrix(RBF_eval_train, train$label)
# Accuracy : 0.9487

# Evaluation on complete train dataset
RBF_eval_test <- predict(RBF_model, test)
# Confusion matrix for RBF Kernel on complete test dataset
confusionMatrix(RBF_eval_test , test$label)
# Accuracy : 0.9489

# The Accuracy is consistent between the whole train and test datasets

##################  Hyperparameter tuning and Cross Validation ################

# The below train function that is used for Cross validation and tuning runs for 
# ~ 25 mins on a Windows 10 64 bit operating system i3 4160 processor at 3.60 GHZ 
# with no other processes running. 

# The train function from caret package enables us to perform Cross Validation. 

# traincontrol function controls the computational nuances of the train function.
#  method =  CV implies Cross Validation.
#  Number =  5 implies Number of folds in Cross Validation.
trainControl <- trainControl(method = "cv", number = 5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric <- "Accuracy"

set.seed(20)
# From the previous RBF Kernel model :
# We know  that C  was selected as 1 . Lets increase C value and give 
# in the range of 1 to 4
# and we know that sigma was selected as 1.62e-7. 
# Lets give the sigma values as 0.62e-7, 1.62e-7., 2.62e-7 
# (i.e -1+sigma, sigma and 1+sigma)
grid <- expand.grid(.sigma = c(0.62e-7, 1.62e-7, 2.62e-7, 3.62e-7),
                    .C = c(1, 2, 3))

# Lets run train function with Method as "svmRadial"  (RBF model), 
# Metric as Accuracy, tuneGrid with the Grid of Parameters already set 
# and trcontrol as traincontrol method.

# There are warnings regarding Scaling of Data. These are due to columns
# where we have no variance for the entire column which cannot be
# scaled. Hence we can ignore these. Lets suppress these warnings
options(warn = -1)
(svm_cv <- train(label~., data = sample_train, 
                 method = "svmRadial", metric = metric, 
                 tuneGrid = grid, trControl = trainControl))

options(warn = 0)

# sigma     C  Accuracy  Kappa    
# 6.20e-08  1  0.9258    0.9175556
# 6.20e-08  2  0.9338    0.9264444
# 6.20e-08  3  0.9370    0.9300000
# 1.62e-07  1  0.9444    0.9382222
# 1.62e-07  2  0.9524    0.9471111
# 1.62e-07  3  0.9528    0.9475556
# 2.62e-07  1  0.9538    0.9486667
# 2.62e-07  2  0.9584    0.9537778
# 2.62e-07  3  0.9590    0.9544444
# 3.62e-07  1  0.9582    0.9535556
# 3.62e-07  2  0.9614    0.9571111
# 3.62e-07  3  0.9616    0.9573333

plot(svm_cv)

###########################    Final model building    #######################

# Lets build our final model with C = 3 and sigma = 3.62e-7
RBF_final_model  <- ksvm(label~ ., data = sample_train, 
                         scaled = F, C = 3, kpar = list(sigma = 3.62e-7),
                         kernel = "rbfdot")

###########################    Final model evaluation   #######################
RBF_final_eval_train   <- predict(RBF_final_model, train)
(conf_train <- confusionMatrix(RBF_final_eval_train, train$label))
# Accuracy : 0.9628

# Confusion Matrix and Statistics
            # Reference
# Prediction#      0    1    2    3    4    5    6    7    8    9
            # 0 5843    1   23    9    4   18   14    9   17   29
            # 1    0 6632   31   13   10    4    9   33   34    9
            # 2   10   27 5729   82   12   13   27   50   45   17
            # 3    3   18   27 5821    0   80    1    8   79   46
            # 4    5   13   29    6 5694   14   31   51   33  120
            # 5   14   14    7   74    3 5228   82    8   84   20
            # 6   23    6   24    3   17   45 5746    0   32    1
            # 7    1    5   43   37    8    1    0 6001   12   78
            # 8   20   14   38   63    5    8    8   10 5464   19
            # 9    4   12    7   23   89   10    0   95   51 5610

# As we saw in EDA, 
# Number label '1' is the most easily distinguishable class 
# because the average intensity is the least compared to other classes. 
# Number label '0' is also easily distinguishable class 
# since the average intensity is very high compared to other classes. 
# Hence both these classes have higher balanced accuracy, sensitivity, 
# specificity, Pos pred Value and Neg Pred Value compared to all other classes.  

# Lets look at the misclassifications: 
#  Most misclassifications in descending order below: 
#  9 misclassified as 4 
#  7 misclassified as 9 
#  4 misclassified as 9 

# Lets look at other metrics : 
colMeans(conf_train$byClass)

RBF_final_eval_test   <- predict(RBF_final_model, test)
(conf_test <-  confusionMatrix(RBF_final_eval_test, test$label))

# Accuracy : 0.9616 

# Confusion Matrix and Statistics
            # Reference
# Prediction#      0    1    2    3    4    5    6    7    8    9
            # 0  967    0    8    1    1    3    7    0    4    8
            # 1    0 1120    1    0    1    0    2   11    0    6
            # 2    1    3  989    7    3    0    1   17    8    2
            # 3    1    3    5  974    0   19    0    4   14   10
            # 4    0    0    5    0  953    1    8    4   10   23
            # 5    5    1    0    8    0  854   10    0    9    3
            # 6    4    5    4    0    5    8  927    0    6    0
            # 7    1    1   10    6    1    1    0  972    4    7
            # 8    1    2   10   11    1    5    3    4  915    5
            # 9    0    0    0    3   17    1    0   16    4  945

#  Most misclassifications in descending order below: 
#  9 misclassified as 4 
#  5 misclassified as 3
#  4 misclassified as 9 , 7 misclassified as 2 
#  7 misclassified as 9

colMeans(conf_test$byClass)


###############################  Final Summary  ############################## 

# Final model is a Non linear model : RBF model (RBF_final_model) 
# after performing Cross validation

# Finalized Hyperparameters : 
    # C     = 3
    # Sigma = 3.62e-07 

# Evaluation metrics on complete Train dataset with 60000 observations 
    # Overall Accuracy  : 96.28%
    # Average values across all classes 
        # Sensitivity          Specificity             
        # 0.9625914            0.9958704  
        # Neg Pred Value       Pos Pred Value     
        # 0.9958706            0.9624732 
        # Recall               F1                Precision  
        # 0.9625914            0.9624654         0.9624732 

# Evaluation metrics on Test dataset with 10000 observations 
    # Overall Accuracy  : 96.16%
    # Average values across all classes 
        # Sensitivity          Specificity       
        # 0.9613238            0.9957356  
        # Neg Pred Value       Pos Pred Value       
        # 0.9957382            0.9613466      
        # Recall               F1                  Precision         
        # 0.9613238            0.9612700           0.9613466     

# The test accuracy is the same as that of the train accuracy indicating 
#  no overfitting or underfitting 
#************************************End of file*******************************