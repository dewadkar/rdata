getwd()

#Set the working directory
?setwd
setwd("R/")

library(help = "datasets")
library(MASS)

# Benign or Malignant

wbcd <- read.csv('train_data/wisc_bc_data.csv', stringsAsFactors=FALSE)

dim(wbcd)
head(wbcd)

# displaying the names of the objects in the data set
names(wbcd)

# displaying the names of the column
colnames(wbcd)
# showing the class of the object
class(wbcd)

# basic summary of the data
summary(wbcd)
# data summary
str(wbcd)
library(ggplot2) # Data visualization
# correlation
round(cor(wbcd[,2:31], use="pair"),2)
# Corrgram of the entire dataset
corrgram(wbcd, order=NULL, lower.panel=panel.shade, upper.panel=NULL, text.panel=panel.txt,
         main="Corrgram of the data")

# another visualization
corrgram(wbcd[,2:10], order=TRUE,
         main="PC2/PC1 order",
         lower.panel=panel.shade, upper.panel=panel.pie,
         diag.panel=panel.minmax, text.panel=panel.txt)

# another visualization
corrgram(wbcd[,2:10], order=TRUE,
         main="correlation ellipses",
         panel=panel.ellipse,
         text.panel=panel.txt, diag.panel=panel.minmax)

corrgram(wbcd[,2:10],
         main=" data with example panel functions",
         lower.panel=panel.pts, upper.panel=panel.conf,
         diag.panel=panel.density)

#histograms
hist(wbcd[,2])

colnames(wbcd)

# Run a scatterplot matrix on the entire dataset
scatterplotMatrix(~radius_mean +
                    texture_mean +
                    perimeter_mean +
                    area_mean +
                    smoothness_mean	+
                    compactness_mean	+
                    concavity_mean	+
                    fractal_dimension_mean	+
                    diagnosis
                  ,data=wbcd, main="Simple Scatterplot Matrix")

# Look at the variable distributions
ggplot(wbcd, aes(x = radius_mean)) +
  geom_bar() +
  ggtitle("Distribution of radius_mean for the entire dataset")

ggplot(wbcd, aes(x = area_mean)) +
  geom_bar() +
  ggtitle("Distribution of area_mean for the entire dataset")



wbcd <- wbcd[-1]
colnames(wbcd)
wbcd$diagnosis <- factor(wbcd$diagnosis, levels=c('B','M'),
                         labels=c('Benign','Malignant'))

colnames(wbcd)
# Model Fitting
# Start off with this (alpha = 0.05)
wbcd$diagnosis <- as.factor(wbcd$diagnosis)
model_algorithm = model <- glm(diagnosis ~ radius_mean +
                                 texture_mean +
                                 perimeter_mean +
                                 area_mean +
                                 smoothness_mean	+
                                 symmetry_mean	+
                                 fractal_dimension_mean	+
                                 area_se ,
                               family=binomial(link='logit'),data=wbcd[1:300,])


# print(summary(model_algorithm))

nrow(wbcd)
colnames(wbcd)
str(wbcd$diagnosis)
# Apply the algorithm to the training sample
prediction_training = predict(model_algorithm,wbcd[1:300,], type = "response")
prediction_training = ifelse(prediction_training > 0.5, 'B', 'M')
error = mean(prediction_training != wbcd$diagnosis)
print(paste('Model Accuracy',1-error))

# Apply the algorithm to the testing sample
prediction_testing = predict(model_algorithm,wbcd[301:500,], type = "response")
prediction_testing = ifelse(prediction_testing > 0.5, 'B', 'M')
error = mean(prediction_testing != wbcd$diagnosis)
print(paste('Model Accuracy',1-error))


