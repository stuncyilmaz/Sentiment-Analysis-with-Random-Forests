# Sentiment-Analysis-with-Random-Forests

Here is an implementation of sentoment analysis using neural networks. I worked with the Rotten Tomatoes dataset from the Kaggle competition.
(https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews). The data contains 156000 short movie reviews, labels with ratings 1-5.

Since it is a large dataset, the algorithm takes some time. I would recommend to create with a smaller data set, using the command line:
head -100 train.tsv>train_small.csv

When you run "main.py" file, "train" and "test" directories are created in the "mydata"  directory. The "train" and "test" directories will contain json files that are used to create the feature matrices for the machine learning algorithms. Executing "main.py" will aslo create "my_kaggle_submission.csv" that contains the output of the random forest classifier. The best cross-validation scores have been achieved with 5 features per tree, and 500 trees (score = 62%).

The main features that the model uses are the tfidf scores corresponding to the 5 rating categories. Each word is assigned a tfidf score as well as a frequency of appearance for each class in the training set.
The features tfidf0..tfidf4 for a given review is the sum of the tfidf scores of the words in the review for the rating class 1...5. Similarly, the columns freq0..4 for a given review is the sum of the frequencies of the words corresponding to the rating class 1...5. Other simple, descriptive features include "exclamation Points", "inner Punctuation", "number of Question Marks", "average WordLength", "number of Words", "is it First Person" etc.

If in the file "predict_testSet.py" the parameter addWords_bool=True, then the top 250 words with the highest training set tfidf scores are included as binary features as well.

The features matrices that are created are very generic, and can be used by any classifier in scit-learn package.
