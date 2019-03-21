''' Data Preprocessing '''

# Dataset
#   - First step is obviously, to import your dataset.
#   - In every dataset, we have two entities - Independent variables and Dependent variables.
#   - Independent variables are used to predict Dependent Variables.
#
# In our current dataset, the first three columns are independent variables and the fourth column is the dependent
# variable which is to be predicted.

# Data Preprocessing - is the first thing to be done after loading your dataset, so that your data is ready for any
# ML algorithm to be applied on it.



#-------------------------------------- Approach -----------------------------------------------------------------
# *** Importing Libraries ***
#   - numpy  -->  This library contains mathematical tools.
#   - matplotlib  -->  This is the library used for data visualisation. (We use, `pyplot` library that is present
#                      inside the matplotlib library - The pyplot library is used for plotting of data.)
#   - pandas  -->  This is the best library to import and manage our dataset.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# *** Setting the `Current-Working-Directory` ***
#   - As we need to import our dataset, if our dataset and the source file (or script) will be in the same folder,
#     and that folder is selected as our current-working-directory then we don't have to provide the full path every
#     time, we need to import some other module or dataset from that directory.
#
# For Spyder on Windows, after saving the python file in the same directory with the dataset, we can run our file
# which will also set the directory in which the file is, as the current-working-directory (Press-`F5` for that).

# ***** Importing the dataset *****
dataset = pd.read_csv('Data.csv')
# After executing our above command in the spyder's IPython Console, we got our variable explorer, in which we can
# see the dataset variable of type-`DataFrame` of size (10, 4) with the Column names separately. Now, if we double-
# click on that, the dataset will open in a new pop-up window. And for our convenience we have another additional
# first column in the display pop-up windows which shows the indexes.

# ***** Making the feature matrix (Independent Vairables) and output vector (Dependent Variable) *****
features = dataset.iloc[:, :-1].values
output = dataset.iloc[:, -1].values
# The values that are returned based on index-location (iloc) are of type numpy.ndarray

# ***** Taking care of `Missing-Data` *****
#   - First approach can be deleting the Lines with the empty data cell. (When we have imported it using pandas
#     then the dataframe data is created will automatically changes the blank cells with `NaN`.)  
#     This approach is little bit danagerous if every row contains some crucial information.
#
#   - Second approach (i.e., the most commonly used approach) is to fill the empty data cell value with the mean
#     of all the values present in that column.
from sklearn.impute import SimpleImputer   
# We are importing the sklearn (scikit-learn) library which contains amazing libraries for Machine Learning.
# Among those amzing libraries, one is the `impute` which helps in the dealing with values of data.
# Inside that library we have the SimpleImputer class which will allow us to take care of the missing data.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  
# By default also, missing_values=np.nan and strategy='mean'.
features[:, 1:3] = imputer.fit_transform(features[:, 1:3])
# Implements the imputer object layout for the specified data values from the specified columns and hence 
# providing the values for the missing data  --> fit.
# Giving the original feature matrix after transforming the missing data with the fitted data --> transform.

# ***** Taking care of `Categorical-Data` *****
#   - Since, Machine learning models are based on Mathematical calculations so, it will cause problems if we have
#     the text data. So firstly, we have to `encode those labels`.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
features_label_encoder = LabelEncoder()
features[: ,0] = features_label_encoder.fit_transform(features[:, 0])
# The numbering given are on the basis of lexographical order.
#
#   - Now the problem of text data is removed but since the ML models are based on Mathematical calculations, and
#     in Mathematics, 2 > 1 > 0, so the ML models will take that 2 has higher priority than the other two. But in
#     real that is not true, as all three of them are just provinding info and not priority and are independent 
#     of each other. For this we are going to use `Dummy Variables`.
onehotencoder = OneHotEncoder(categorical_features=[0])
features = onehotencoder.fit_transform(features).toarray()     # toarray() - converts whole matrix of one built-in type.
# After providing the Dummy Variables the original column will be divided into number of columns equivalent to
# the categories present. Each column representing one of those categories.
# In Our Example: First col --> France, Second col --> Germany and Third col --> Spain, i.e., based on lexical
# ordering.
#
# Doing the same for output vector:
output_label_encoder = LabelEncoder()
output = output_label_encoder.fit_transform(output)
# We do not need to onehotencode it, as it is the dependent variable for which we are going to predict the output, then
# it will gonna know it that it is categorical data.
 
# ***** Splitting the Dataset - Train and Test *****
#   - Since we are letting the Machine Learn (i.e., ML algo), so we need to check that whether that machine is learning or
#     or too much learning (craming) by testing the learning of that machine on a slightly different dataset. For that
#     purpose, we are going to split our dataset into two parts - training dataset and the testing dataset.
#   - Machine is going to learn on the training dataset and then we are gonna test it on the testing dataset.
from sklearn.model_selection import train_test_split
training_features, testing_features, training_ouput, testing_output = train_test_split(features, output, test_size=0.2, 
                                                                                       random_state=0)

# ***** Feature-Scaling *****
#   - Here in our case, we have two columns Age and Salary having the numerial data. But the range of these data vary a
#     lot. And since many ML algo work on the euclidean's distance --> sqrt((x2-x1)**2 + (y2-y1)**2), where say, we have
#     two points P1(x1, y1) and P2(x2, y2) where Age denotes x-coordinates and Salary denotes y-coordinate, so since the
#     range value of Salary is very large the output will be dominated by the Salary which will be equivalent to the case
#     where we can say that, feature salary is having a high priority than the feature age that is not the case right now.
#
# So, in order to remove that unintentional effect, we are gonna scale our features. We have following ways:
#   - Standardisation  -->  stand_x = (x - mean(x)) / stand_deviation(x)
#   - Normalisation  -->  norm_x = (x-min(x)) / max(x) - min(x)
from sklearn.preprocessing import StandardScaler
features_standard_scaler = StandardScaler()
training_features = features_standard_scaler.fit_transform(training_features)
testing_features = features_standard_scaler.transform(testing_features)     # we don't need to fit again for test set.
# For interpretition, we will not scale the dummy variables when doing the same for ML algos for now, but for now we can
# do it, as we don't have any need of dummy_variable interpretition right now.
# Even if some of the ML algos is not based on euclidean distance (like - Decision tress) still feature scaling helps in
# converging much faster.
# In this case, the dependent variable is a categorical data (for classification problems) so we need not to scale it,
# but in case of regression - dependent variable there will be a need for scaling.
