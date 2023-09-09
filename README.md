# Disaster-Tweets
 
Predicting Disaster Tweets using Natural Processing Language

## Abstract
It is a challenging task to actively acquire information of distress calls and emergency situations in disaster-affected area in a timely manner. In such situation social media can play a dominant role. With its widespread network, Twitter text analysis can help authorities to respond to such situation using information from the real-time tweets shared by thousands of individuals. In this paper, we propose a machine learning algorithm 

## Introduction
In recent times Twitter has been categorized as one of the most powerful channels to share information. Millions of communications happen on a daily basis allowing people to express their social, political or spiritual views. With such a widespread use and large amounts of data shared, it becomes necessary to validate the truth in every tweet especially in case of disaster emergencies. During emergencies, tweets can help disaster management agencies, state police and first-responders take prompt action. In this paper, we design a machine learning model that helps to differentiate tweets that correspond to actual emergencies vs random tweets. We evaluate the model by using different model parameters also incorporating measures to avoid overfitting. 

## Data exploration
For this project, we used the Twitter dataset provided by Kaggle. The dataset is divided into train set containing 7613 tweets with 5 column parameters, and test set containing 3263 tweets with 4 parameters, excluding target variable. For our model, the target is defined as a binary variable, indicating 1 if the tweet is a disaster and 0 otherwise. The distribution of target class shows that data is slightly imbalanced towards the 0 label. However, it not severe enough to affect the final classifier. We now analyze each of the predictor variables.
Keyword
This variable seems to provide a keyword summary for every tweet. For the raw train set, some of the most common keywords for tweets are ‘Fatalities’, ‘Armageddon’, ‘Sinking’, ‘Evacuate’ etc. This field can be a prime parameter while building an algorithm that predicts the type of emergency for a tweet categorized as a disaster (not in this report scope).
Location
This variable helps identify the location of a tweet. Although the column contains approx. 33% missing data, we also observe many misclassifications, for example, tweets talking about California Wildfires have location as Worldwide. Certain location values contain integer values. Also, majority of the tweets originate from states within the United States indicating a sampling bias. Furthermore, many data inconsistencies can be observed, for instance some tweets use abbreviations such as NY/ NYC/ NY state to indicate New York. To avoid such issues, we standardize them by identifying location by country instead of city/state. This analysis shows UK and India with the highest tweet count following US. 
Text
The text captures the description or content of the tweet. This is a main variable used in our model to classify if a tweet is a disaster or not. Further analysis on the text field show certain patterns such text with long word count tend to be not disasters. We use word clouds as they give a good interpretation of the commonly used words in disaster tweets.



Fig. 1.  Need to put a figure!!
## Data processing
Prior to performing any text modelling, we verify if there are any duplicate row values in the train set. Having duplicates could cause the model to mis-classify results. There are 52 such entries based on the column’s 'keyword', 'location', 'text', 'target'; Hence we keep only one and drop the other duplicates. 

Another set of tweets (<0.1%) have the same text but are classified differently with respect to the target value. This could be due to some error while data capturing. Hence, we manually assign the appropriate target labels to each of these tweets. 

As described in the previous section, for our model development, we use only the ‘text’ column of each tweet to classify if the tweet is a disaster or not. Prior to providing this as a model input, we perform text pre-processing to remove any non-contributing content.
Convert all text into common case(lower) to allow the model to interpret them similarly
Removing Hashtags ‘#’ as they are the most commonly used letter in tweets thus not adding any value
Removing any html tags, urls or hyperlinks within the tweet
Removing emoticon tags
Tweets often contain stock market value tags
Removing any re-tweet or punctuation signs
Removing @

Additionally, we also use NLTK stop_words dictionary to remove the most commonly used stopwords such as ‘the’, ‘is’, ‘an’, ‘who’ etc. These words do not add any additional information to the text, hence can be removed. We also incorporate some more words into the stop_words list such that are more relevant to our dataset.

These steps of pre-processing are done simultaneously for both train and test data ensuring that a uniform text field is given as model input. 

## Model prediction
As the next step, we input this pre-processed field to build various different models, and test each model using 5-fold stratified cross-validation. The reason for using a stratified approach is because we observed a slight imbalance in our dataset. This approach will shuffle the dataset, prior to splitting, into 5-folds ensuring the ratio of observations in each class remains the same.

Prior to model building, we convert the text data to a numerical vector representation that can be fed as an input to the model. This can be done by 
Count Vectorization: Represents a matrix with each cell showing the frequency count of a specific term in a given document
TF-IDF score represents the relative importance of a term in the document and the entire corpus. It consists of two components: the normalized Term Frequency (TF), and the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.

The next step in the text classification framework is to train a classifier using the features created. We evaluate model performance against each of the below models for both tf-idf and count_vector.
Logistic Regression
Logistic regression measures the relationship between target dependent variable and one or more independent variables i.e. ‘text’ field, by estimating probabilities using a logistic/sigmoid function. This is most commonly used classier for any text classification.
### SVM
Support Vector Machine follows a geometric interpretation approach. The model extracts a best possible hyper-plane / line that segregates the two classes. The main advantage of SVM is that compared complex algorithms like neural networks they have better speed and performance when the sample size is in thousands.
### Naïve Bayes
Here we used multinomial Naïve Bayes classifier for the classification of tweets. This algorithm is based upon Bayes theorem which assumes that every element is independent of one another. Thus, each feature is classified irrespective of other features. Absence of a particular feature will not affect the others. Naïve Bayes works well when the corpus is small such as tweets. It calculates the probability of each tag for a given sample and returns the tag with the highest probability as output. In this case, whether it is a disaster or not.
### Stochastic Gradient Decent
This classification algorithm creates decision tree on the data samples and collects prediction from every tree and ultimately chooses the best by voting.
### Random Forest
This classification algorithm creates decision tree on the data samples and collects prediction from every tree and ultimately chooses the best by voting. By averaging the result, it does reduces the overfitting of data.
### LightGBM
Light Gradient Boosting Machine is mainly used to handle large amounts of data. It is a distributed gradient boosting framework for machine learning. It is based on decision tree algorithms and used to classify whether our tweets indicate disaster or not. It occupies less memory and its main focus is on getting results with good accuracy hence gradient boosting can lead to overfitting the data if there is a large amount of noisy data.
### LSTM
Long Short-Term Memory networks is a RNN based model. It learns order dependance in sequence prediction problems. It can memorize information for a longer period compared to RNN. It works well on the twitter dataset as the text is short and the task is simple classification.

On running each model with 5-fold cross-validation, we take the average of each model's performance on all 5 folds and use it as an indicator to decide the best performing model for our test data.


Fig. 2.  Need to put a figure!!


Fig. 3.  Need to put a figure!!


### Handling model overfitting
Overfitting occurs when a model doesn't generalize well and doesn't perform well on unknown dataset. Overfitting is avoided using using various ways. 

Layer removal technique is applied which forces the model to learn only the important patterns that matter and hence the ones that minimizes the loss. This helped in reducing undefitting in the model. 

Another method used is addition of droput layers which sets the output as 0 randomly for a layer. This helps slow down the model obverfitting. 

L2 regularization is performed on the model thus adding cost to the loss function for having large weight. This forces the model to learn only relevant patterns in the train data.

### References
References are important to the reader; therefore, each citation must be complete and correct. There is no editorial check on references; therefore, an incomplete or wrong reference will be published unless caught by a reviewer or discusser and will detract from the authority and value of the paper. References should be readily available publications.

Basic format for books:
J. K. Author, “Title of chapter in the book,” in Title of His Published Book, xth ed. City of Publisher, Country if not USA: Abbrev. of Publisher, year, ch. x, sec. x, pp. xxx–xxx.
Examples:
G. O. Young, “Synthetic structure of industrial plastics,” in Plastics, 2nd ed., vol. 3, J. Peters, Ed. New York: McGraw-Hill, 1964, pp. 15–64.
W.-K. Chen, Linear Networks and Systems. Belmont, CA: Wadsworth, 1993, pp. 123–135.

Basic format for periodicals:
J. K. Author, “Name of paper,” Abbrev. Title of Periodical, vol. x, no. x, pp. xxx–xxx, Abbrev. Month, year.
Examples:
J. U. Duncombe, “Infrared navigation—Part I: An assessment of feasibility,” IEEE Trans. Electron Devices, vol. ED-11, no. 1, pp. 34–39, Jan. 1959.
E. P. Wigner, “Theory of traveling-wave optical laser,” Phys. Rev., vol. 134, pp. A635–A646, Dec. 1965.
E. H. Miller, “A note on reflector arrays,” IEEE Trans. Antennas 

Statement of collaboration
This project is a team contribution of members including Anusha Arun Kumar, Tanya Shourya, Eeshapriya Gutta

