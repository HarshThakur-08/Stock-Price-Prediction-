# Stock-Price-Prediction-
## Berkshire Hathaway stocks have been forcasted in this project.

# Motivation:
Artificial Intelligence and Machine Learning is the need of the hour. It is said to be the
new Electricity in the sense that it will transform industries the way Electricity did about
100 years ago. Thus, we must gain an in-depth knowledge of this field so that we are
able to apply our knowledge and expertise in the real world. So, I have selected this
problem which will help us to achieve just that. Stock Price Prediction can be a very
great tool to compare various Machine Learning and Deep Learning algorithms and can
provide us with an in-depth understanding of each of the algorithms used. 


# Objective:
The main objective of this project is to predict and forecast the value of the stocks of a
company. The recent trend in this domain has been of using effective Machine Learning
and Deep Learning techniques to make predictions on the current values of company
stocks by training on their historical values.

# Idea:
Predictions can be made based on several lines such as physical vs. psychological
factors, rational vs. irrational behavior, and so on. All these aspects combine to form
volatile share prices making them very difficult to predict with a high degree of
accuracy.
Using features such as the latest announcements about a corporation, their quarterly
revenue results and so on, Machine Learning and Deep Learning techniques have shown
the potential to unearth patterns and insights never seen before, making them useful to
make accurate predictions.
I will be implementing an amalgam of Machine Learning and Deep Learning
algorithms for our project. I will start with simple algorithms such as averaging and
linear regression, and then gradually move on to advanced techniques such as LSTM
and Auto ARIMA.

# Experimental Design:
## Datasets:
The dataset being utilized for analysis was picked up from Yahoo Finance. The dataset
consisted of 1259 records of the required stock prices and other relevant values. The data
reflected the stock prices at certain time intervals for each day of the year. It consisted of
various attributes namely date, open, high, low, close, adj close and volume. For the
purpose of simulation and analysis, the data for only one company (Berkshire Hathaway
Inc. Class B (BRK.B)) was considered. All the data was available in a file of csv format
which was first read and transformed into a data-frame using the Pandas library in Python.
Following this normalization of the data was performed through usage of the sklearn
library in Python and the data was divided into training and testing sets. The validation
set was kept as 17% of the available dataset.
## Research Variables:
The target variable was taken as ‘Close’ which indicates the final price at which the stock
is traded on a particular day.
For every technique the predictors were chosen as required by the technique.
## Techniques or Methods used:
Throughout the project I have used various Machine Learning techniques to predict
the stock prices of a company based on historic values. The Algorithms implemented till
now are:
1)Moving Average
2)Linear Regression
3)k–Nearest Neighbours (k-NN)
4)Auto ARIMA
5)Prophet
6)Long Short Term Memory (LSTM)
## Statistical Test:
The performance of all the Machine Learning models was assessed using Chi-square
test for goodness-of-fit of models.


# Techniques used for stock price prediction:
## A. Moving Average
‘Average’ is easily one of the most common things we use in our day-to-day lives. For
instance, calculating the average marks to determine overall performance, or finding the
average temperature of the past few days to get an idea about today’s temperature – these
all are routine tasks we do on a regular basis. So this is a good starting point to use on our
dataset for making predictions.
The predicted closing price for each day will be the average of a set of previously
observed values. Instead of using the simple average, we will be using the moving average
technique which uses the latest set of values for each prediction.

<img width="362" alt="image" src="https://user-images.githubusercontent.com/70577185/178911774-72cb9d6b-507b-42ee-907a-74623f1d8714.png">

## B. Linear Regression
The most basic machine learning algorithm that can be implemented on this data is
linear regression. The linear regression model returns an equation that determines the
relationship between the independent variables and the dependent variable.
The equation for linear regression can be written as:

Y = w0X0 + w1X1 + w2X2 + w3X3 +... +wnXn

Where w’s are weights and X’s are dependent variables.

<img width="349" alt="image" src="https://user-images.githubusercontent.com/70577185/178911956-eec892f0-0fe2-479a-8cf5-f434ae7507f3.png">

## C. k-Nearest Neighbours
K-nearest neighbor technique is a machine learning algorithm that is considered as
simple to implement. The stock prediction problem can be mapped into a similarity based
classification. The historical stock data and the test data is mapped into a set of vectors.
Each vector represents the N dimension for each stock feature.
Then, a similarity metric such as Euclidean distance is computed to take a decision.
KNN is considered a lazy learning that does not build a model or function previously, but
yields the closest k records of the training data set that have the highest similarity to the
test (i.e. query record).
Then, a majority vote is performed among the selected k records to determine the class
label and then assigned it
to the query record.
The prediction of stock market closing price is computed using kNN as follows:
a) Determine the number of nearest neighbors, k.
b) Compute the distance between the training samples and the query record.
c) Sort all training records according to the distance values.
d) Use a majority vote for the class labels of k nearest neighbors, and assign it as a
prediction value of the query record.

## D. Auto Arima
ARIMA models take into account the past values to predict the future values. There are
three important parameters in ARIMA:
p = (past values used for forecasting the next value)
q = (past forecast errors used to predict the future values)
d = (order of differencing)
Parameter tuning for ARIMA consumes a lot of time. So we will use auto ARIMA
which automatically selects the best combination of (p,q,d) that provides the least error.

## E. Prophet
The input for Prophet is a dataframe with two columns: date and target (ds and y).
Prophet tries to capture the seasonality in the past data and works well when the dataset
is large.

## F.LSTM
LSTMs are widely used for sequence prediction problems and have proven to be
extremely effective. The reason they work so well is because LSTM is able to store past
information that is important, and forget the information that is not. LSTM has three
gates:

● The input gate: The input gate adds information to the cell state

● The forget gate: It removes the information that is no longer required by the
model

● The output gate: Output Gate at LSTM selects the information to be shown as
output.

Since stock market involves processing of huge data, the gradients with respect to the
weight matrix may become very small and may degrade the learning rate.[8].This
corresponds to the problem of Vanishing Gradient. LSTM prevents this from happening.
The LSTM consists of a remembering cell, input gate, output gate and a forget gate. The
cell remembers the value for long term propagation and the gates regulate them.

<img width="349" alt="image" src="https://user-images.githubusercontent.com/70577185/178912215-41ad3a64-fb95-40f5-978b-79f5bf82d865.png">

# Statistical Test used:
The Chi-square goodness of fit test is a statistical hypothesis test used to determine
whether a variable is likely to come from a specified distribution or not. It is often used
to evaluate whether sample data is representative of the full population.
The Chi-square goodness of fit test checks whether your sample data is likely to be from
a specific theoretical distribution. I have a set of data values, and an idea about how
the data values are distributed. The test gives us a way to decide if the data values have
a “good enough” fit to our idea, or if our idea is questionable.
I used it to evaluate the goodness-of-fit of various Machine Learning models on the
dataset.

# Results:
The proposed system is trained and tested over the dataset taken from Yahoo Finance. It
is split into training and testing sets respectively and yields the following results upon
passing through the different models:
## A. Moving Average
RMSE- 64.55786109999444

<img width="672" alt="image" src="https://user-images.githubusercontent.com/70577185/178912597-a40815a5-1003-4d0c-a46b-45cfce7972a7.png">

The RMSE value is close to 64.5 but the results are not very promising (as you can
gather from the plot). The predicted values are of the same range as the observed values
in the train set (there is an increasing trend initially and then a slow decrease).

## B. Linear Regression
RMSE- 50.63619732621797

<img width="672" alt="image" src="https://user-images.githubusercontent.com/70577185/178912821-ae665080-d9f7-44e2-9906-7652491730d3.png">

Linear regression is a simple technique and quite easy to interpret, but there are a few
obvious disadvantages. One problem in using regression algorithms is that the model
overfits to the date and month column. Instead of taking into account the previous
values from the point of prediction, the model will consider the value from the same
date a month ago, or the same date/month a year ago.

## C. k-Nearest Neighbours (KNN)
RMSE- 99.76651715131152

<img width="672" alt="image" src="https://user-images.githubusercontent.com/70577185/178912992-bfbf48f9-37d9-4ed9-881f-91227a7d15fd.png">

## D. Auto ARIMA
RMSE- 14.677147246175059
As we saw earlier, an auto ARIMA model uses past data to understand the pattern in the
time series. Using these values, the model captured an increasing trend in the series.
Although the predictions using this technique are far better than that of the previously
implemented machine learning models, these predictions are still not close to the real
values.

<img width="672" alt="image" src="https://user-images.githubusercontent.com/70577185/178913095-69464957-7368-4ae1-bdfe-0273372db7b2.png">

As its evident from the plot, the model has captured a trend in the series, but does not
focus on the seasonal part.

## E. Prophet
RMSE-53.77768613976217
Prophet (like most time series forecasting techniques) tries to capture the trend and
seasonality from past data. This model usually performs well on time series datasets, but
fails to live up to it’s reputation in this case.

<img width="672" alt="image" src="https://user-images.githubusercontent.com/70577185/178913220-3ac2868d-1d2c-451e-b9cd-553b334bcf3d.png">

As it turns out, stock prices do not have a particular trend or seasonality. It highly
depends on what is currently going on in the market and thus the prices rise and fall.
Hence forecasting techniques like ARIMA, SARIMA and Prophet would not show good
results for this particular problem.

## LSTM
RMSE-3.935778491845232

<img width="672" alt="image" src="https://user-images.githubusercontent.com/70577185/178913334-65acdc34-21af-4f78-aec5-43be53124305.png">

The LSTM model can be tuned for various parameters such as changing the number of
LSTM layers, adding dropout value or increasing the number of epochs.

## G. Statistical Test
Evaluating Performance using Statistical Test (Chi-squared Test)

● To test the goodness-of-fit of models.

Ho: There is no significant difference in the performance of the algorithms.
Ha: There is a significant difference in the performance of the algorithms.
Chi-square value: 126.24825285783061
p-value: 1.4875638451412125e-25
There is a significant difference in the performance of the algorithms (reject the Null
Hypothesis)
Thus, I can conclude that the algorithm - Long Short Term Memory (LSTM) performs
the best on the given dataset; as there is a significant difference in the performance of
the algorithms and LSTM has the lowest RMSE value out of all the algorithms.

<img width="221" alt="image" src="https://user-images.githubusercontent.com/70577185/178913644-cf14d961-abdb-43c8-85b3-a42d1ed2524e.png">

# Conclusion and Future work:
As from the results we can see that LSTM has proven to be more reliable for stock
market price prediction , the RMSE value is very less in the LSTM model thus error rate
is very less as compared to other machine learning algorithms. But the predictions from
LSTM are not enough to identify whether the stock price will increase or decrease.
Stock price is affected by the news about the company and other factors like
demonetization or merger/demerger of the companies. There are certain intangible
factors as well which can often be impossible to predict beforehand. In the future, the
accuracy of the stock market prediction system can be further improved by utilizing a
much bigger dataset than the one being utilized currently. Furthermore, other emerging
models of Machine Learning could also be studied to check for the accuracy rate
resulting from them. Sentiment analysis though Machine Learning on how news
affects the stock prices of a company is also a very promising area. Other deep learning
based models can also be used for prediction purposes.
I further aim to analyze and implement the following two SCIE journal papers:
1) Stock closing price prediction based on sentiment analysis and LSTM
2) CNN-BiLSTM-AM method for stock price prediction
Both have been published by the journal Neural Computing and Applications.
