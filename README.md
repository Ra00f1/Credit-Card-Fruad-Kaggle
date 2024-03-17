# Introduction
This project was made to explore and test Credit Card Fraud Detection Dataset from Kaggle(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) which was offered as a choice in one of my course homework’s in university to explore and explain.
## Data
At first glance it is obvious that most of the data except the Time and Amount columns are scaled already and the impact of scaling or not scaling those two columns seems minimal from my testing’s.
## Methods & Results
Like the last project I also tried a variety of methods, but I also thought that Decision trees would work well for this dataset which was proved after adding and testing it as it had the most accuracy however it was nearly identical to the accuracy of Logistic regression.

Upon deeper analysis Decision tree also has better scores in recall and f1-score compared to Logistic regression but I also noticed some of the projects at Kaggle using Cross-validation to see which one performs better which is defiantly a good idea to be sure which one is better.

The most challenging part of this project was visualizing the dataset:

  •	  The large amount of the dataset (284K rows and 32 columns each)  was too much for my device.

  •	  Because most of the dataset is already scaled it is very hard to extract any meaningful information from the graphs at my skill level.
