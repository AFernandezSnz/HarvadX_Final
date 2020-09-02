

##0.Dataset and Packages downloading 
    ##Packages Download
    if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
    if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
    if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
    
    library(tidyverse)
    library(caret)
    library(lattice)
    library(ggplot2)
    library(data.table)
    library(dplyr)
    library("readr")

    # Online Shoppers Purchasing Intention Dataset, csv downloaded via:
    #http://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
    #http://archive.ics.uci.edu/ml/machine-learning-databases/00468/
  
    #Creating data-set:
    getwd() 
    data<-read.csv("online_shoppers_intention (1).csv")
    attach(data)

##1. Exploring data-set:
str(data)
summary(data)
table(data$Revenue)

  #From logical to chr : Revenue

  data <- data %>% mutate(Revenue = replace(Revenue, Revenue == "FALSE","KO"))
  data <- data %>% mutate(Revenue = replace(Revenue, Revenue == "TRUE","GOOD"))
  class(data$Revenue)

  #Predictor's models names
  predictors = names(training)[names(training) != "Revenue"]
  predictors 
  
##2.Creating training-test dataset
  n=nrow(data)
  ind=1:n
  itraining=sample(ind,floor(n*0.85))
  itest=sample(setdiff(ind,itraining),floor(n*0.15))
  training = data[itraining,]
  testing = data[itest,]
  dim(training)
  dim(testing)

  ##Balancing data-set:
  downSampled_training = downSample(x=training[, -ncol(training)],
                                    y=as.factor(training$Revenue))
  downSampled_testing = downSample(x=testing[, -ncol(testing)],
                                   y=as.factor(testing$Revenue))



##3.SVM  Linear Model
    ##3.1. Unbalanced Data: Training svm unbalanced data. Fiting the model-> C parameter
    train_control<-trainControl(method="cv", number = 10, p = .9)
    svm_fit_u_l <- train(Revenue ~ ., method = "svmLinear", data = training, 
                     trControl = train_control, 
                     tuneGrid = expand.grid(C = seq(0.01, 0.2, length = 10)))
    plot(svm_fit_u_l)
    # The best tuning parameter C that maximizes model accuracy
    c<-svm_fit_u_l$bestTune
    results_acc_by_c_u_l<-as_tibble(svm_fit_u_l$results[which.max(svm_fit_u_l$results[,2]),])
    results_acc_by_c_u_l
    
    # Applying best C
    svm_def_u_l <- train(Revenue ~ ., method = "svmLinear", data = training, 
                     trControl = train_control, 
                     cost= c)
    # LinearModel Accuracy for Unbalanced Data
    y_hat_svm <- predict(svm_def_u_l, testing, type = "raw")
    confusion_matrix_u_l<-confusionMatrix(y_hat_svm, as.factor(testing$Revenue))
    overall_Ac_u_l<-confusion_matrix_u_l$overall[["Accuracy"]]
    Sensitivity_Ac_u_l<-confusion_matrix_u_l$byClass[["Sensitivity"]]
    Specificity_Ac_u_l <-confusion_matrix_u_l$byClass[["Specificity"]]
    Pos_pred_value_u_l<-confusion_matrix_u_l$byClass[["Pos Pred Value"]]
    Neg_pred_value_u_l<-confusion_matrix_u_l$byClass[["Neg Pred Value"]]
    f_measure_u_l <- 2 * ((Pos_pred_value_u_l * Specificity_Ac_u_l) / (Pos_pred_value_u_l+ Specificity_Ac_u_l))
    cat(sprintf("Overall Accuracy=%.2f , Sensitivity = %.3f,  Specificity = %.3f,F = %.3f, PPV = %.3f\n, NPV = %.3f\n",
                overall_Ac_u_l, Sensitivity_Ac_u_l, Specificity_Ac_u_l,f_measure_u_l,Pos_pred_value_u_l,Neg_pred_value_u_l))
    confusion_matrix_u_l$table
    
    
    ##3.2. Balanced Data:SVM
    
      ##Training svm balanced data. Fiting the model-> C parameter
      svm_fit_b_l <- train(Class ~ ., method = "svmLinear", data = downSampled_training, 
                       trControl = train_control, 
                       tuneGrid = expand.grid(C = seq(0.01, 0.2, length = 10)))
      plot(svm_fit_b_l)
      # The best tuning parameter C that maximizes model accuracy
      cbl<-svm_fit_b_l$bestTune
      results_acc_by_c_b_l<-as_tibble(svm_fit_b_l$results[which.max(svm_fit_b_l$results[,2]),])
      results_acc_by_c_b_l
      
      # Aplying best C
      svm_def_b_l <- train(Class ~ ., method = "svmLinear", data = downSampled_training, 
                       trControl = train_control, cost=cbl)
      # LinearModel Accuracy for Balanced Data
      y_hat_svm_b_l <- predict(svm_def_b_l, downSampled_testing, type = "raw")
      Confusion_Matrix_b_l<-confusionMatrix(y_hat_svm_b_l, as.factor(downSampled_testing$Class))
      Overall_Ac_b_l<-Confusion_Matrix_b_l$overall[["Accuracy"]]
      Sensitivity_Ac_b_l<-Confusion_Matrix_b_l$byClass[["Sensitivity"]]
      Specificity_Ac_b_l <-Confusion_Matrix_b_l$byClass[["Specificity"]]
      Pos_pred_value_b_l<-Confusion_Matrix_b_l$byClass[["Pos Pred Value"]]
      Neg_pred_value_b_l<-Confusion_Matrix_b_l$byClass[["Neg Pred Value"]]
      f_measure_b_l <- 2 * ((Pos_pred_value_b_l * Specificity_Ac_b_l) / (Pos_pred_value_b_l+ Specificity_Ac_b_l))
      cat(sprintf("Overall Accuracy=%.2f , Sensitivity = %.3f,  Specificity = %.3f,F = %.3f, PPV = %.3f\n, NPV = %.3f\n",
                  Overall_Ac_b_l, Sensitivity_Ac_b_l, Specificity_Ac_b_l,f_measure_b_l,Pos_pred_value_b_l,Neg_pred_value_b_l))
      Confusion_Matrix_b_l$table
      
      
    ##3.2 Random Forest
      ##3.2.1. Training RF unbalanced data. 
      train_control<-trainControl(method="cv", number = 10, p = .9)
      svm_fit_u_r <- train(Revenue ~ ., method = "rf", data = training, tuneLength=5,
                       trControl = train_control)
      plot(svm_fit_u_r)
      # The best tuning mtry that maximizes model accuracy
      mtry_u<-svm_fit_u_r$bestTune
      results_acc_mtry_u_r<-as_tibble(svm_fit_u_r$results[which.max(svm_fit_u_r$results[,2]),])
      results_acc_mtry_u_r
      
      # Applying best mtry
      svm_def_u_r <- train(Revenue ~ ., method = "rf", data = training, minNode=mtry_u$mtry,
                           trControl = train_control)
      
      # RF Accuracy for unbalanced data
      y_hat_svm_u_r <- predict(svm_def_u_r, testing, type = "raw")
      confusion_matrix_u_r<-confusionMatrix(y_hat_svm_u_r, as.factor(testing$Revenue))
      Overall_Ac_u_r<-confusion_matrix_u_r$overall[["Accuracy"]]
      Sensitivity_Ac_u_r<-confusion_matrix_u_r$byClass[["Sensitivity"]]
      Specificity_Ac_u_r <-confusion_matrix_u_r$byClass[["Specificity"]]
      Pos_pred_value_u_r<-confusion_matrix_u_r$byClass[["Pos Pred Value"]]
      Neg_pred_value_u_r<-confusion_matrix_u_r$byClass[["Neg Pred Value"]]
      f_measure_u_r <- 2 * ((Pos_pred_value_u_r * Specificity_Ac_u_r) / (Pos_pred_value_u_r + Specificity_Ac_u_r))
      cat(sprintf("Overall Accuracy=%.2f , Sensitivity = %.3f,  Specificity = %.3f,F = %.3f, PPV = %.3f\n, NPV = %.3f\n",
                  Overall_Ac_u_r, Sensitivity_Ac_u_r, Specificity_Ac_u_r,f_measure_u_r,Pos_pred_value_u_r,Neg_pred_value_u_r))
      confusion_matrix_u_r$table
      
    ##3.2. Training RF balanced data.
      
      ##Training svm balanced data. Fiting the model-> C parameter
      svm_fit_b_r <- train(Class ~ ., method = "rf", data = downSampled_training, 
                           tuneLength=5,
                           trControl = train_control)
      plot(svm_fit_b_r)
      # The best tuning mtry that maximizes model accuracy
      cbr<-svm_fit_b_r$bestTune
      results_acc_by_c_rb<-as_tibble(svm_fit_b_r$results[which.max(svm_fit_b_r$results[,2]),])
      results_acc_by_c_rb
      
      # Aplying best mtry
      svm_def_b_r <- train(Class ~ ., method = "rf", data = downSampled_training, 
                           minNode=cbr$mtry,
                           trControl = train_control,)
      #RF Accuracy for balanced data
      y_hat_svm_b_r <- predict(svm_def_b_r, downSampled_testing, type = "raw")
      Confusion_Matrix_b_r<-confusionMatrix(y_hat_svm_b_r, as.factor(downSampled_testing$Class))
      Overall_Ac_b_r<-Confusion_Matrix_b_r$overall[["Accuracy"]]
      Sensitivity_Ac_b_r<-Confusion_Matrix_b_r$byClass[["Sensitivity"]]
      Specificity_Ac_b_r <-Confusion_Matrix_b_r$byClass[["Specificity"]]
      Pos_pred_value_b_r<-Confusion_Matrix_b_r$byClass[["Pos Pred Value"]]
      Neg_pred_value_b_r<-Confusion_Matrix_b_r$byClass[["Neg Pred Value"]]
      f_measure_b_r <- 2 * ((Pos_pred_value_b_r * Specificity_Ac_b_r) / (Pos_pred_value_b_r + Specificity_Ac_b_r))
      cat(sprintf("Overall Accuracy=%.2f , Sensitivity = %.3f,  Specificity = %.3f,F = %.3f, PPV = %.3f\n, NPV = %.3f\n",
                  Overall_Ac_b_r, Sensitivity_Ac_b_r, Specificity_Ac_b_r,f_measure_b_r,Pos_pred_value_b_r,Neg_pred_value_b_r))
      Confusion_Matrix_b_r$table
##Results   
    ##Results Overall Acuracy
        models_gen_accuracy <- data_frame(Model=c("SVM-Unbalanced","SVM-Balanced","RF-Unbalanced","RF-Balanced"), 
                                     Accuracy = c(overall_Ac_u_l,Overall_Ac_b_l,Overall_Ac_u_r,Overall_Ac_b_r))
        models_gen_accuracy%>%knitr::kable(format="rst")
    
    ##Results  Specificity
        models_spec <- data_frame(Model=c("SVM-Unbalanced","SVM-Balanced","RF-Unbalanced","RF-Balanced"), 
                                          Specificity = c(Specificity_Ac_u_l,Specificity_Ac_b_l,Specificity_Ac_u_r,Specificity_Ac_b_r))
        models_spec%>%knitr::kable(format="rst")
        
    ##Results  Sensitivity
        models_sensitivity <- data_frame(Model=c("SVM-Unbalanced","SVM-Balanced","RF-Unbalanced","RF-Balanced"), 
                                         Sensitivity = c(Sensitivity_Ac_u_l,Sensitivity_Ac_b_l,Sensitivity_Ac_u_r,Sensitivity_Ac_b_r))
        models_sensitivity%>%knitr::kable(format="rst")
        
    ##Results TP
        models_tp <- data_frame(Model=c("SVM-Unbalanced","SVM-Balanced","RF-Unbalanced","RF-Balanced"), 
                                         TPrate= c(Pos_pred_value_u_l,Pos_pred_value_b_l,Pos_pred_value_u_r,Pos_pred_value_b_r))
        models_tp%>%knitr::kable(format="rst")   
    
    ##Results TN
        models_tn <- data_frame(Model=c("SVM-Unbalanced","SVM-Balanced","RF-Unbalanced","RF-Balanced"), 
                                         TNrate= c(Neg_pred_value_u_l,Neg_pred_value_b_l,Neg_pred_value_u_r,Neg_pred_value_b_r))
        models_tn%>%knitr::kable(format="rst")    
    ##Results F
        f_measure <- data_frame(Model=c("SVM-Unbalanced","SVM-Balanced","RF-Unbalanced","RF-Balanced"), 
                                F_measure= c(f_measure_u_l,f_measure_b_l,f_measure_u_r,f_measure_b_r))
        f_measure%>%knitr::kable(format="rst")    
        
    
cbind(models_gen_accuracy,models_sensitivity[ , 2:2],models_spec[ , 2:2],models_tn[ , 2:2],models_tp[ , 2:2],f_measure[ , 2:2])%>%knitr::kable()   
