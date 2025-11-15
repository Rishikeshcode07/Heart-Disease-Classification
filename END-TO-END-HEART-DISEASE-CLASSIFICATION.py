

# # Predicting heart disease using machine learning
# 
# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
# 
# We are going to take the following approach:
# 
# 1. Problem Definition
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling
# 6. Experimentation
# 
# ## 1. Problem Defination
# 
# In a statement,
# > Given clinical parameters about a patient, can we predict whether or not they have heart disease ?
# 
# 
# ## 2. Data
# 
# The original data came from the Cleavland data from the UCI Machine Learning Repository.
# Link:- https://archive.ics.uci.edu/dataset/45/heart+disease
# 
# 
# There is also a version of it available on Kaggle.
# Link:- https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# 
# 
# ## 3. Evaluation
# 
# > If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.
# 
# ## 4. Features
# 
# This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).
# 
# **Create data dictionary**
# 
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
# 
#     *  0: Typical angina: chest pain related decrease blood supply to the heart
#     *  1: Atypical angina: chest pain not related to heart
#     *  2: Non-anginal pain: typically esophageal spasms (non heart related)
#     *  3: Asymptomatic: chest pain not showing signs of disease
# 
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything   above 130-140 is typically cause for concern.
# 5. chol - serum cholestoral in mg/dl
# 
#     * serum = LDL + HDL + .2 * TRIGLYSERIDES
#     * above 200 is cause for concern
# 
# 6. fbs -(fasting blood sugar > 120 mg/dl) (1 = true: 0 = false)
# 
#     * '>126' mg/dl signals diabetes
# 
# 7. restech -resting electrocardiographic results
# 
#     * 0: Nothing to note
#     * 1: ST-T Wave abnormality
#         * can range from mild symptoms to severe problems
#         * signals non-normal heart beat
#     * 2. Possible or definite left ventricular hypertrophy
#         * Enlarged heart's main pumping chamber
#      
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
# 11. slope - the slope of the peak exercise ST segment
#     * 0: Upsloping: better heart rate with excercise (uncommon)
#     * 1: Flatsloping: minimal change (typical healthy heart)
#     * 2: Downslopins: signs of unhealthy heart
#       
# 12. ca - number of major vessels (0-3) colored by flourosopy
#     
#      * colored vessel means the doctor can see the blood passing through
#      * the more blood movement the better (no clots)
# 
# 13. thal - thalium stress result
#     
#     * 1,3: normal
#     * 6: fixed defect: used to be defect but ok now
#     * 7: reversable defect: no proper blood movement when excercising
# 
# 14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)    


# ## Preparing the tools 
# 
# We're going to use pandas , matplotlib and Numpy for data analysis and manipulation. 


# Import all the tools we need

# Regula EDA( Exploratory Data Analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# we want our plots to appear inside the notebook
%matplotlib inline  

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay

# ## Load the data


df = pd.read_csv("heart-disease.csv")
df.shape  # rows and columns

# ## Data Exploration (exploratory data analysis or EDA)
# 
# The goal hear is to find out more about the data and become a subject matter export on the dataset you're working with. 
# 
# 1. What question(s) are you trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. What's missing from the data and how do you deal with it?
# 4. Where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?


df.head()

# Let's find out how many of each class there
df["target"].value_counts()

## This will show how many of them have heart disease and how many of them don't have .

# As you can see they are almost the same or we can say that they are almost balance, so we can say that it is balance class classification.


df["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"]);

# We can also the different information about this datasets.
df.info()

# Are there any missing values
df.isna().sum()

# **Now we will compaier the columns with each other so that we can start gaining an intuition about how the features are related to the target variables.**


# **So what we will do is that we will take the 1st two column and compaier with the target label.**


# ## Heart Disease Frequency according to Sex


df.sex.value_counts()

# By seeing this we can say that , our data is more tillted towards male than the femails.
# That means, male are more with heart disease than the femails.


# Compare target column with sex column
pd.crosstab(df.target, df.sex)

# what this information is saying that , in femails out of 96 , 72 femails have heart disease and 24 don't have. 
# 
# In male, out of 207 males, 114 males don't have have heart disease and 93 have heart disease.


# Create a plot of crosstab
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                    figsize=(10, 6),
                                    color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Diesease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"]);
plt.xticks(rotation=0);

# **Next comparison is between age vs heart rate**
# 
# ## Age vs. Max Heart Rate for Heart Disease


df["thalach"].value_counts()

# **So this thalach have 91 different values to init so this bar grafh might not be the best way to see the data . We can use scatter plot for this to visualize different types of pattern in it.**


# Create another figure
plt.figure(figsize=(10, 6))

# Scatter with postivie examples
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);

# Check the distribution of the age column with a histogram
df.age.plot.hist();

# ### Heart Disease Frequency per Chest Pain Type
# 
# 3. cp - chest pain type
#     * 0: Typical angina: chest pain related decrease blood supply to the heart
#     * 1: Atypical angina: chest pain not related to heart
#     * 2: Non-anginal pain: typically esophageal spasms (non heart related)
#     * 3: Asymptomatic: chest pain not showing signs of disease


pd.crosstab(df.cp, df.target)

# Make the crosstab more visual
pd.crosstab(df.cp, df.target).plot(kind="bar",
                                   figsize=(10, 6),
                                   color=["salmon", "lightblue"])

# Add some communication
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation=0);

# Make a correlation matrix
# Correlation matrix says aout how each independent variable is related to each other.
# Or me can say how these columns interact with each other.
df.corr()

# Let's make our correlation matrix a little prettier
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# **So now what to do next is model driven EDA. So what that means is building a machine learning model to dtive insights of how these independent variables here contribute to the target varible.  Now correlation matrix is one of them , but we're more intrested in trying to predict for the future.**


# ## 5. Modelling


# Split data into X and y
X = df.drop("target", axis=1)

y = df["target"]

X

y

# Split data into train and test sets
np.random.seed(42)

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

X_train

y_train, len(y_train)

# Now we've got our data split into training and test sets, it's time to build a machine learning model.
# 
# We'll train it (find the patterns) on the training set.
# 
# And we'll test it (use the patterns) on the test set.
# 
# We're going to try 3 different machine learning models:
# 
#     1. Logistic Regression
#     2. K-Nearest Neighbours Classifier
#     3. Random Forest Classifier
# 
# **We are using Logistic Regression because despite its name , is a linear model for classification rather than regression.**   


# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)

model_scores

# ### Model Comparison


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();

# Now we've got a baseline model... and we know a model's first predictions aren't always what we should based our next steps off. What should we do?
# 
# Let's look at the following:
# 
#     * Hypyterparameter tuning
#     * Feature importance
#     * Confusion matrix
#     * Cross-validation
#     * Precision
#     * Recall
#     * F1 score
#     * Classification report
#     * ROC curve
#     * Area under the curve (AUC)


# ### Hyperparameter Tuning (by hand)


# Let's tune KNN

# So what we will do is that , we will take two list ,because we want to do 
# is compare different versions of the same model, so model with different
# settings and compare their scores on the two different data sets.

train_scores = []
test_scores = []

# Create a list of differnt values for n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores list
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores list
    test_scores.append(knn.score(X_test, y_test))

# So now what this is going to do is that , this is going to do is just
# going to loop through this range of 1 to 20 and then it's going to create
# 20 different K and N models and append their scores to list.

train_scores

test_scores

plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")

# ### Hyperparameter tuning with RandomizedSearchCV
# 
# We're going to tune:
# 
#     * LogisticRegression()
#     * RandomForestClassifier()
# ... using RandomizedSearchCV


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

# Now we've got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV...


# Tune LogisticRegression

np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)

rs_log_reg.best_params_

# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()...


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)

# Find the best hyperparameters
rs_rf.best_params_

# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(X_test, y_test)

# ### Hyperparamter Tuning with GridSearchCV
# 
# Since our LogisticRegression model provides the best scores so far, we'll try and improve them again using GridSearchCV..


# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train);

# Check the best hyperparmaters
gs_log_reg.best_params_

# Evaluate the grid search LogisticRegression model
gs_log_reg.score(X_test, y_test)

# ### Evaluting our tuned machine learning classifier, beyond accuracy
# 
# * ROC curve and AUC score
# * Confusion matrix
# * Classification report
# * Precision
# * Recall
# * F1-score
#   
# ... and it would be great if cross-validation was used where possible.
# 
# To make comparisons and evaluate our trained model, first we need to make predictions.


# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)

y_preds

y_test

# **The ROC curve is created by plotting the true positive rate against the false positive rate at various thresold settings.**


# Plot ROC curve and calculate and calculate AUC metric
RocCurveDisplay.from_estimator(gs_log_reg, X_test, y_test)

# Confusion matrix
print(confusion_matrix(y_test, y_preds))

sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
plot_conf_mat(y_test, y_preds)

# Now we've got a ROC curve, an AUC metric and a confusion matrix, let's get a classification report as well as cross-validated precision, recall and f1-score.


print(classification_report(y_test, y_preds))

# ### Calculate evaluation metrics using cross-validation
# 
# We're going to calculate accuracy, precision, recall and f1-score of our model using cross-validation and to do so we'll be using cross_val_score().


# Check best hyperparameters
gs_log_reg.best_params_

# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc

cv_acc = np.mean(cv_acc)
cv_acc

# Cross-validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision=np.mean(cv_precision)
cv_precision

# Cross-validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall

# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1

# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                      legend=False);

# ### Feature Importance
# 
# Feature importance is another as asking, "which features contributed most to the outcomes of the model and how did they contribute?"
# 
# Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".
# 
# Let's find the feature importance for our LogisticRegression model...


# Fit an instance of LogisticRegression
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

clf.fit(X_train, y_train);

# Check coef_
clf.coef_

# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict

# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);

pd.crosstab(df["sex"], df["target"])

pd.crosstab(df["slope"], df["target"])

# slope - the slope of the peak exercise ST segment
# 
# * 0: Upsloping: better heart rate with excercise (uncommon)
# * 1: Flatsloping: minimal change (typical healthy heart)
# * 2: Downslopins: signs of unhealthy heart


# ## SUMMARY OF THIS PROJECT
# 
# Objective: Predict heart disease presence using clinical data from the UCI Cleveland heart disease dataset.
# 
# Dataset: 303 samples with 13 clinical features including age, sex, chest pain, blood pressure, cholesterol, and exercise-induced angina; balanced classes with slightly more male patients.
# 
# Exploratory Data Analysis:
# 
#     * Data checked for missing values and class balance.
# 
#     * Visualized feature distributions and correlations.
# 
#     * Noted males have higher heart disease incidence than females.
# 
# Models Used: Logistic Regression, K-Nearest Neighbors, Random Forest Classifier.
# 
# Methodology:
# 
#     * Data split into train (80%) and test (20%) sets.
# 
#     * Models trained and evaluated with accuracy scores.
# 
# Results:
# 
#     * Logistic Regression achieved ~88.5% accuracy.
# 
#     * Random Forest and KNN scored ~83.6% and ~68.9% respectively.
# 
#     * Hyperparameter tuning improved Random Forest accuracy to ~86.9%.
# 
# Tools: Python libraries including numpy, pandas, matplotlib, seaborn, scikit-learn, and Jupyter Notebook.
# 
# Key Takeaway: Demonstrated end-to-end machine learning pipeline for binary classification with interpretable medical data.
# 
# Outcome: Confirmed Logistic Regression as best model with solid predictive performance though further improvements possible.