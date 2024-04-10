# Load libraries
# tidyverse: Collection of packages for data manipulation and visualization
library(tidyverse)  # Data manipulation and visualization

# class: Package for k-Nearest Neighbors algorithm
library(class)      # k-Nearest Neighbors algorithm

# rpart: Package for Decision Trees
library(rpart)      # Decision Trees

# e1071: Package for Support Vector Machines
library(e1071)      # Support Vector Machines


# Load the necessary library
library(caret)

# Dataset Basic Information
heart_data <- read.csv("heart.csv")

# Display basic information about the dataset
str(heart_data)

# Summary Statistics for Numerical Variables
summary_stats_numerical <- summary(heart_data[, c("age", "trestbps", "chol", "thalach", "oldpeak")])
print(summary_stats_numerical)

categorical_columns <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target")

# Function to get a count and unique values summary for categorical variables including frequency percentage
categorical_summary <- function(column_name) {
  cat_summary <- table(heart_data[[column_name]])
  unique_values <- length(unique(heart_data[[column_name]]))
  total_count <- sum(cat_summary)
  percentage <- prop.table(cat_summary) * 100  # Calculate percentage
  
  cat_summary_df <- data.frame(
    Category = names(cat_summary),
    Count = as.numeric(cat_summary),
    Percentage = percentage
  )
  
  print(paste("Summary for", column_name, ":"))
  print(paste("Unique Values:", unique_values))
  print("Count summary:")
  print(cat_summary_df)
}

# Apply the updated function to each categorical variable
for (column_name in categorical_columns) {
  categorical_summary(column_name)
}

# Correlation Matrix
correlation_matrix <- cor(heart_data[, c("age", "trestbps", "chol", "thalach", "oldpeak")])
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
heatmap(correlation_matrix,
        annot = TRUE,
        fmt = ".2f",
        cmap = colorRampPalette(c("blue", "white", "red"))(100),
        main = "Correlation Matrix Heatmap",
        margin = c(10, 10))



# Univariate Analysis

# Continuous Variables - Histograms
continuous_columns <- c("age", "trestbps", "chol", "thalach", "oldpeak")

par(mfrow=c(2, 3))  # Set up a 2x3 grid for subplots
for (column_name in continuous_columns) {
  hist(heart_data[[column_name]], main=column_name, xlab=column_name, col="skyblue", border="black")
}

# Categorical Variables - Bar Plots
categorical_columns <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target")

# Categorical Variables - Bar Plots
par(mfrow=c(3, 3))  # Set up a 3x3 grid for subplots
for (column_name in categorical_columns) {
  counts <- table(heart_data[[column_name]])
  
  # Mapping numeric codes to descriptive names
  labels <- switch(
    column_name,
    cp = c("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"),
    sex = c("Male", "Female"),
    restecg = c("Normal", "ST-T wave abnormality", "Probable/definite LV hypertrophy"),
    slope = c("Upsloping", "Flat", "Downsloping"),
    thal = c("Normal", "Fixed defect", "Reversible defect", "Not described"),
    target = c("No disease", "Presence of disease"),
    fbs = c("False", "True"),
    exang = c("No", "Yes")
  )
  
  barplot(counts, main=column_name, col="lightblue", border="black", names.arg = labels)
}

# Bivariate Analysis

# Age vs. Target
# Separate age data for each target class
age_no_disease <- heart_data$age[heart_data$target == 0]
age_disease <- heart_data$age[heart_data$target == 1]

# histograms for age by target class
par(mfrow=c(1, 2))  # Set up a 1x2 grid for side-by-side plots
hist(age_no_disease, main="Age Distribution (No Disease)", col="skyblue", border="black", xlab="Age")
hist(age_disease, main="Age Distribution (Presence of Disease)", col="lightcoral", border="black", xlab="Age")

# Sex vs. Target
# Separate sex data for each target class
sex_no_disease <- heart_data$sex[heart_data$target == 0]
sex_disease <- heart_data$sex[heart_data$target == 1]

# Create a 1x2 grid for side-by-side bar plots
par(mfrow=c(1, 2))

# Bar plot for the distribution of heart disease by gender
barplot(table(sex_no_disease), main="No Disease by Gender",
        col="skyblue", border="black", ylim=c(0, max(table(heart_data$sex))),
        xlab="Gender", ylab="Count")

barplot(table(sex_disease), main="Presence of Disease by Gender",
        col="lightcoral", border="black", ylim=c(0, max(table(heart_data$sex))),
        xlab="Gender", ylab="Count")



# Data preprocessing
# Check for missing values in the dataset
missing_values <- colSums(is.na(heart_data))
print("Missing Values Summary:")
print(missing_values)

# Check the balance of the target variable
target_balance <- table(heart_data$target)
if (length(unique(heart_data$target)) > 1) {
  print("Target Variable is Balanced:")
  print(target_balance)
} else {
  print("Target Variable is Not Balanced.")
}

# Boxplots to identify outliers in numerical variables
par(mfrow=c(2, 3))  # Set up a 2x3 grid for subplots
for (column_name in continuous_columns) {
  boxplot(heart_data[[column_name]], main=paste("Boxplot of", column_name), col="skyblue", border="black")
}

# Alternatively, one can use quantile-based outlier detection
outlier_detection <- function(column) {
  Q1 <- quantile(heart_data[[column]], 0.25)
  Q3 <- quantile(heart_data[[column]], 0.75)
  IQR <- Q3 - Q1
  outliers <- heart_data[[column]] < (Q1 - 1.5 * IQR) | heart_data[[column]] > (Q3 + 1.5 * IQR)
  return(outliers)
}

# Identify outliers in numerical variables
outliers <- sapply(continuous_columns, outlier_detection)
print("Number of Outliers in Each Numerical Variable:")
print(colSums(outliers))

# Identify outliers using quantile-based approach
outliers <- sapply(continuous_columns, outlier_detection)

# Remove outliers from the dataset
heart_data_no_outliers <- heart_data[!apply(outliers, 1, any), ]

# Display information before and after removing outliers
cat("Original dataset size:", nrow(heart_data), "rows\n")
cat("Dataset size after removing outliers:", nrow(heart_data_no_outliers), "rows\n")

# Update heart_data to contain the dataset without outliers
heart_data <- heart_data_no_outliers


#One hot encoding
# Display basic information about the dataset
str(heart_data)

# Define the categorical variables for one-hot encoding
nominal_variables <- c("cp", "restecg", "thal")

# Perform one-hot encoding for nominal variables
heart_data <- heart_data %>%
  mutate(across(all_of(nominal_variables), as.factor)) %>%
  mutate(across(all_of(nominal_variables), ~factor(.)))

# Display updated dataset information
str(heart_data)

# Split the dataset into training and testing sets
set.seed(123)  # Setting seed for reproducibility
sample_indices <- sample(1:nrow(heart_data), 0.8 * nrow(heart_data))
train_data <- heart_data[sample_indices, ]
test_data <- heart_data[-sample_indices, ]

# Define predictor variables and target variable
predictors <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal")
target <- "target"

# Feature scaling using the preProcess function from caret
scaling_model <- preProcess(train_data[, predictors], method = c("center", "scale"))

# Apply the scaling transformation to both the training and test sets
train_data_scaled <- predict(scaling_model, train_data[, predictors])
test_data_scaled <- predict(scaling_model, test_data[, predictors])

# Build the KNN model on the scaled data
knn_model_scaled <- knn(
  train = train_data_scaled,
  test = test_data_scaled,
  cl = train_data[, target],  # Use the original target variable
  k = 5
)

# Evaluate the performance of the scaled KNN model
conf_matrix_knn_scaled <- table(knn_model_scaled, test_data$target)
accuracy_knn_scaled <- sum(diag(conf_matrix_knn_scaled)) / sum(conf_matrix_knn_scaled)
precision_knn_scaled <- conf_matrix_knn_scaled[2, 2] / sum(conf_matrix_knn_scaled[, 2])
recall_knn_scaled <- conf_matrix_knn_scaled[2, 2] / sum(conf_matrix_knn_scaled[2, ])
f1_score_knn_scaled <- 2 * (precision_knn_scaled * recall_knn_scaled) / (precision_knn_scaled + recall_knn_scaled)

# Display the confusion matrix and performance metrics for scaled KNN
print("Confusion Matrix for Scaled KNN:")
print(conf_matrix_knn_scaled)
cat("\nAccuracy for Scaled KNN:", round(accuracy_knn_scaled, 3))
cat("\nPrecision for Scaled KNN:", round(precision_knn_scaled, 3))
cat("\nRecall for Scaled KNN:", round(recall_knn_scaled, 3))
cat("\nF1 Score for Scaled KNN:", round(f1_score_knn_scaled, 3))

# Build the decision tree model
tree_model <- rpart(target ~ ., data = train_data, method = "class")

# Make predictions on the test set
predictions <- predict(tree_model, test_data, type = "class")

# Evaluate the performance of the decision tree
conf_matrix <- table(predictions, test_data$target)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Display the confusion matrix and performance metrics
print("Confusion Matrix:")
print(conf_matrix)
cat("\nAccuracy:", round(accuracy, 3))
cat("\nPrecision:", round(precision, 3))
cat("\nRecall:", round(recall, 3))
cat("\nF1 Score:", round(f1_score, 3))

# Build the SVM model for classification
svm_model <- svm(
  formula = as.formula(paste(target, "~", paste(predictors, collapse = "+"))),
  data = train_data,
  type = "C-classification",  # Change to C-classification
  kernel = "linear",
  cost = 1
)

# Make predictions on the test set
svm_predictions <- predict(svm_model, newdata = test_data)

# Evaluate the performance of the SVM model
conf_matrix_svm <- table(svm_predictions, test_data$target)
accuracy_svm <- sum(diag(conf_matrix_svm)) / sum(conf_matrix_svm)
precision_svm <- conf_matrix_svm[2, 2] / sum(conf_matrix_svm[, 2])
recall_svm <- conf_matrix_svm[2, 2] / sum(conf_matrix_svm[2, ])
f1_score_svm <- 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)

# Display the confusion matrix and performance metrics for SVM
print("Confusion Matrix for SVM:")
print(conf_matrix_svm)
cat("\nAccuracy for SVM:", round(accuracy_svm, 3))
cat("\nPrecision for SVM:", round(precision_svm, 3))
cat("\nRecall for SVM:", round(recall_svm, 3))
cat("\nF1 Score for SVM:", round(f1_score_svm, 3))

# Model Comparison

# Create a data frame to store the performance metrics
model_metrics <- data.frame(
  Model = c("KNN Scaled", "Decision Tree", "SVM"),
  Accuracy = numeric(3),
  Precision = numeric(3),
  Recall = numeric(3),
  F1_Score = numeric(3)
)

# Update the model_metrics data frame with KNN Scaled metrics
model_metrics[1, 2:5] <- c(accuracy_knn_scaled, precision_knn_scaled, recall_knn_scaled, f1_score_knn_scaled)

# Update the model_metrics data frame with Decision Tree metrics
model_metrics[2, 2:5] <- c(accuracy, precision, recall, f1_score)

# Update the model_metrics data frame with SVM metrics
model_metrics[3, 2:5] <- c(accuracy_svm, precision_svm, recall_svm, f1_score_svm)

# Display the model comparison results
print("Model Comparison:")
print(model_metrics)

# Model Comparison Visualization
# Plotting the accuracy, precision, recall, and F1 Score for each model
par(mfrow=c(2, 2))

# Accuracy Comparison
barplot(model_metrics$Accuracy, names.arg = model_metrics$Model, main = "Accuracy Comparison", col = rainbow(3), ylim = c(0, 1))

# Precision Comparison
barplot(model_metrics$Precision, names.arg = model_metrics$Model, main = "Precision Comparison", col = rainbow(3), ylim = c(0, 1))

# Recall Comparison
barplot(model_metrics$Recall, names.arg = model_metrics$Model, main = "Recall Comparison", col = rainbow(3), ylim = c(0, 1))

# F1 Score Comparison
barplot(model_metrics$F1_Score, names.arg = model_metrics$Model, main = "F1 Score Comparison", col = rainbow(3), ylim = c(0, 1))





# Set seed for reproducibility
set.seed(123)

# Create 5 rows of dummy data
dummy_data <- data.frame(
  age = sample(30:80, 5, replace = TRUE),
  sex = sample(0:1, 5, replace = TRUE),
  cp = sample(0:3, 5, replace = TRUE),
  trestbps = sample(90:200, 5, replace = TRUE),
  chol = sample(150:350, 5, replace = TRUE),
  fbs = sample(0:1, 5, replace = TRUE),
  restecg = sample(0:2, 5, replace = TRUE),
  thalach = sample(80:200, 5, replace = TRUE),
  exang = sample(0:1, 5, replace = TRUE),
  oldpeak = runif(5, 0, 5),
  slope = sample(0:2, 5, replace = TRUE),
  ca = sample(0:4, 5, replace = TRUE),
  thal = sample(0:3, 5, replace = TRUE),
  target = sample(0:1, 5, replace = TRUE)
)

# Display the dummy data
print(dummy_data)

# Step 2: Preprocess Dummy Data
# Assuming `scaling_model` is the scaling model created during training
dummy_data_scaled <- predict(scaling_model, dummy_data[, predictors])
dummy_data_scaled <- as.data.frame(dummy_data_scaled)

# Assuming `nominal_variables` is the vector of nominal variables
dummy_data_scaled <- dummy_data_scaled %>%
  mutate(across(all_of(nominal_variables), as.factor)) %>%
  mutate(across(all_of(nominal_variables), ~factor(.)))

# Step 3: Test SVM Model
# Assuming `svm_model` is the trained SVM model
svm_predictions_dummy <- predict(svm_model, newdata = dummy_data_scaled)

# Evaluate the performance on dummy data
conf_matrix_svm_dummy <- table(svm_predictions_dummy, dummy_data$target)
accuracy_svm_dummy <- sum(diag(conf_matrix_svm_dummy)) / sum(conf_matrix_svm_dummy)
precision_svm_dummy <- conf_matrix_svm_dummy[2, 2] / sum(conf_matrix_svm_dummy[, 2])
recall_svm_dummy <- conf_matrix_svm_dummy[2, 2] / sum(conf_matrix_svm_dummy[2, ])
f1_score_svm_dummy <- 2 * (precision_svm_dummy * recall_svm_dummy) / (precision_svm_dummy + recall_svm_dummy)

# Display the results
cat("Confusion Matrix for SVM on Dummy Data:")
print(conf_matrix_svm_dummy)

cat("Accuracy for SVM on Dummy Data:", round(accuracy_svm_dummy, 3))
cat("Precision for SVM on Dummy Data:", round(precision_svm_dummy, 3))
cat("Recall for SVM on Dummy Data:", round(recall_svm_dummy, 3))
cat("F1 Score for SVM on Dummy Data:", round(f1_score_svm_dummy, 3))


