# Multilayer Perceptron Prediction Model

###########################################################
##
##		IMPORT MODULES
##
###########################################################
import csv
import numpy
import pandas
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense


inputFile = 'TractorData.csv'
predTrendFile = 'prediction.csv'

# Number of months to be forecasted
forecastMonths = 24

# fix random seed for reproducibility
numpy.random.seed(7)

###########################################################
##
##		READ DATE AND VALUES SEPARATELY
##
###########################################################
print('Reading input file...')
# open the file in universal line ending mode
with open(inputFile, 'rU') as infile:
  # read the file as a dictionary for each row ({header : value})
  reader = csv.DictReader(infile)
  data = {}
  for row in reader:
    for header, value in row.items():
      try:
        data[header].append(value)
      except KeyError:
        data[header] = [value]

# extract the variables you want
year_month = data['Month']
sales = data['Sales']

# print('series values: ')
# for i in range(len(year_month)):
# 	print('series values: %s' % year_month[i])

YearList = [i.split('-', 1)[0] for i in year_month]
YearSetList = list(set(YearList))
YearSetList.sort()
MonthList = [i.split('-', 1)[1] for i in year_month]
print()

###########################################################


###########################################################
##
##		IDENTIFY YEARLY TREND
##
###########################################################
sumPerYear=[0.0] * len(YearSetList)

print('Identifying yearly trend...')
for i in range(len(YearSetList)):
	for j in range(len(year_month)):
		if(YearSetList[i] in year_month[j]):
			sumPerYear[i]+=float(sales[j])

print('Yearly Trend:')
print(' Year,  Total')
for i in range(len(YearSetList)):
	print(' %s, %f' % (YearSetList[i], sumPerYear[i]))
print()

###########################################################


###########################################################
##
##		LOAD DATA
##
###########################################################
print('Loading data for Time Series Analysis...')
# load the dataset
# dataframe = pandas.read_csv('../resources/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataframe = pandas.read_csv(inputFile, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

###########################################################


###########################################################
##
##		SPLIT DATA INTO TRAIN AND TEST SETS
##
###########################################################
# split into train and test sets
train_size = int(len(dataset) - forecastMonths)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

###########################################################


###########################################################
##
##		FUNCTION:
##		CONVERTS AN ARRY OF VALUES INTO A DATASET MATRIX
##
###########################################################
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)):
		if i == len(dataset)-look_back :
			a = dataset[i]
			dataX.append(a)
			dataY.append(dataset[i])
		else :
			a = dataset[i:(i+look_back), 0]
			dataX.append(a)
			dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

###########################################################


###########################################################
##
##		RESHAPE THE DATA INTO X=1 AND Y=t+1
##
###########################################################
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

###########################################################


###########################################################
##
##		BUILD PREDICTION MODEL
##
###########################################################
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

###########################################################


###########################################################
##
##		ESTIMATE MODEL PERFORMANCE
##
###########################################################
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

###########################################################


###########################################################
##
##		FORECAST
##
###########################################################
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

###########################################################

# shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
# plt.plot(dataset)
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)


###########################################################
##
##		STRORE PREDICTED TREND IN FILE
##
###########################################################
print('Storing predicted trend in the file...')
f = open(predTrendFile, 'wt')
try:
    writer = csv.writer(f)
    writer.writerow( ('Date', 'Predicted', 'Expected') )
    for i in range(len(testPredict)):
        writer.writerow( (year_month[len(year_month)-forecastMonths+i],
						  float(testPredict[i]), float(testX[i])) )
finally:
    f.close()

print("Prediction Trend sotred in File: %s " %predTrendFile)
print()

###########################################################


###########################################################
##
##		PLOT FORECASTED DATA
##
###########################################################
plt.plot(testX)
plt.plot(testPredict)
plt.title("Actual Vs. Forecasted Values")
plt.savefig("NN_Forecast_complete.png", format = 'png')
plt.show()

###########################################################