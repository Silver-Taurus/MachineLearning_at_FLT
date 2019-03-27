''' Natural Language Processing (NLP) '''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ***** Importing the dataset *****
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# ***** Cleaning the texts *****
# The following steps should be taken for cleaning the text:
#   Step1: Removing Non-Ascii Characters (replacing it with `space`)
#   Step2: Getting the lowercase of the line
#   Step3: Getting the words out of line
#   Step4: Removing the stopwords
#   Step5: Suffix Stripping (for getting the `stem` words)
#   Step6: Getting the line back from the words, separated by space
#   Step7: Iterate the above processes for each review and form a corpus
#
# If the `stopwords` is not present in nltk, then:
#   import nltk
#   nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
    ps = PorterStemmer()
    review = ' '.join([ps.stem(word) for word in review if not word in set(stopwords.words('english'))])
    corpus.append(review)
    
# ***** Creating the Bag-Of-Words Model *****
# The first big step of natural language processing is not only that we cleaned all the reviews but we also created a corpus.
# Now we are going to form the Bag-Of-Words Model in which for every one of the 1000 lines will have the 0 or 1 value for the
# cloumns which comprises of words from the corpus. This will show whether that word appears or not in the line. But for this
# purpose we need to get the unique set of words so that the unneeded redundancy reduces, as for our case we are forming a 
# matrix, so adding only one redundant column for 1000 lines will be a lot of waste, so we are going for unique words.
# Now even for unique words we are having a lots of zero in our matrix since not every word will appear frequently in each
# line or review, hence we are going to get a huge sparse matrix. Sparse Matrix can be easily reduced by changing the 
# max_features arguments in CountVectorizer or by using Dimnesionality Reduction Tecgniques.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
bag_of_words = cv.fit_transform(corpus).toarray()
target = dataset.iloc[:, 1].values

# Naive-Bayes and the Random-Forest are two most commonly used algos for NLP. Right now, we are going to use Naive Bayes.

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
trained_features, test_features, trained_target, test_target = train_test_split(bag_of_words, target, test_size=0.2, 
                                                                                random_state=0)

# Fitting Naive Bayes classifier to our Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(trained_features, trained_target)

# Predicting the Test set results
predicted_target = classifier.predict(test_features)

# Making the `Confusion Matrix`
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_target, predicted_target)
