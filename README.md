### Using Patient Features to Detect Diabetes

**Andrew Tseng**

#### Executive summary

The purpose of this project is to identify which features about an individual are the most valuable towards predicting risk of diabetes. This will be done by training and tuning five classificatino models that will be evaluated on whether they can properly detect diabetes based on an indvidiaul's provided features. Each model's performance will be evaluated and compared with each other to determine which model is the best among them. From here, we will take the best model and further determine which features about an individual are the most valuable for predicting diabetes. 

#### Rationale

The project is hoping to solve the issue of growing rates of diabetes in the population. Predicting diabetes aligns with the broader goals of preventive medicine, improving individual and public health outcomes while reducing societal and economic burdens. If caught early enough (pre-diabetic stage), people can make changes to individua lifestyles and potentially prevent progression to diabetes. 

#### Research Question

The question the project is attempting to answer is what are the features about a person that are the most important towards predicting diabetes, as well as determining which classification model can perform this task the best.

#### Data Sources

The data used for this project was sourced from Kaggle at the following link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
The dataset has 100,000 samples with no null values and a distribution of about 90% of the entries not having diabetes and about 10% of the entries having diabetes. Because of this difference in distribution, it was decided that the recall score would be the main metric used for evaluating the models.

![nondiabetic_diabetic](https://github.com/user-attachments/assets/499ef7d5-de90-497a-9e58-5c52dec90aa4)

The only cleaning/preparation necessary for the dataset was converting the values of the 'gender' and 'smoking_history' columns from categorical values to numerical values. This was performed by using the get_dummies() function. The columns that were created as a result were then added to the original dataset, with the original 'gender' and 'smoking_history' columns being removed, resulting in a dataset with only numerical values.

Here are the first five rows for the dataset pre-cleaning.

![pre_cleaning](https://github.com/user-attachments/assets/3c0e04f6-f034-4eee-a678-2ae2065274ff)

#### Methodology

Grid Search Cross Validation was utilized to determine the optimal hyperparameters for each model to maximize the metrics used for evaluation. These metrics are accuracy, precision, and recall score, with recall score being the main metric used for evaluation. It is calculated as (True Positive) / (True Positives + False Negatives), or: 

$$
\frac{TP}{TP + FN}
$$

Five models were trained, tuned, and compared with each other to determine the best model for predicting diabetes. 

**Logistic Regression:** A pipeline object is created to standardize the data using StandardScaler and instantiate a Logistic Regression model. 
GridSearchCV is used to find 3 hyperparameters: the regularization strength (C) with the options being [0.01, 0.1, 1, 10, 100],
what solver to use, which could be ['liblinear', 'saga'], and the penalty, with the options being ['l1', 'l2'].
The optimal model has a regularization strength of 0.01, the solver 'liblinear', and the 'l1' penalty.

**Decision Tree Classifier:** A pipeline object is created to standardize the data using StandardScaler and instantiate a Decision Tree Classifier model. 
GridSearchCV is used to find 4 hyperparameters: the criterion with the options being ['gini', 'entropy'],
the max depth with the options being  [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], the minimum samples split with the options being [1, 2, 3, 4, 5, 10],
and the minimum samples leaf with the options being [1, 2, 3, 4, 5].
The optimal model has the criterion 'gini', max depth of 2, minimum samples leaf of 1, and minimum samples split of 2.

**Random Forest Classifier:** A pipeline object is created to standardize the data using StandardScaler and instantiate a Random Forest Classifier model. 
GridSearchCV is used to find 5 hyperparameters: the number of estimators with the options being [50, 100, 200],
the max depth which could be [None, 10, 20, 30], the minimum samples split with the options being [2, 5, 10], 
the minimum samples leaf with the options being [1, 2, 4], and the boostrap with the options being [True, False].
The optimal model has 200 estimators, a max depth of None, 10 minimum samples split, 4 minimum samples leaf, and bootstrap set to True.

**SVC:** A pipeline object is created to standardize the data using StandardScaler and instantiate a SVC model. 
GridSearchCV is used to find 3 hyperparameters: the regularization parameter (C) with the options being [0.1, 1, 10],
the kernel type which could be ['linear', 'rbf'], and the kernel coefficient (gamma) which could be ['scale', 'auto', 0.1, 1]
The optimal model has a regularization paramter of 10, a kernel type of 'rbf', and a gamma of 0.1.

**KNearestNeighbor:** A pipeline object is created to standardize the data using StandardScaler and instantiate a KNearestNeighbor model. 
GridSearchCV is used to find 4 hyperparameters: the number of neighbors with could range from [1-20] with an interval of 2,
the weight which could be ['uniform', 'distance'], the leaf size with the options being [5, 10, 15], 
and the value p with the options being [1, 2].
The optimal model has 9 neighbors, a weight of 'uniform', a leaf size of 10, and a value of 2 for p.

#### Results
After evaluating all the models, the best model for predicting diabetes was the Random Tree Classifier Model, with an accuracy of 0.9738, a precision score of 0.99656, and a recall score of 0.67857. This performance is then followed by the Decision Tree Classifier model, Logistic Regression model, SVM model, and KNearestNeighbor model. This decision was made by evaluating the accuracy, precision, and recall of each model, although the recall score had more weight over the other metrics due to the nature of the topic. This is because it's more important being able to accurately predict true positives then the overall accuracy.

![overall_results](https://github.com/user-attachments/assets/d3ed7b5f-a915-4d91-8def-435aac84a2fe)

![accuracy_score](https://github.com/user-attachments/assets/2449f0e8-ba31-4a0d-af6d-33ca18e823c2)

![precision_score](https://github.com/user-attachments/assets/6116e135-0e40-49c3-9c19-cf66730bf676)

![recall_score](https://github.com/user-attachments/assets/c3281ce1-5d16-4f54-8829-90279390d5d7)

#### Next steps

It was valuable to determine which features among the ones provided by the dataest are more important when predicting diabetes. However, there are some other features that I would also like to test, such as a person's country of origin and medical history (not just limited to smoking/heart disease). In addition, I would like to further specify certain features. For example, the blood-glucose levels in the dataset had a wide range from 80-300. However, these values are affected by whether the individual has been fasting or recently ate, so having an additional column/feature where the patients are asked how whether they ate within 2 hours of having their blood drawn would assist with providing more detailed glucose levels.

Additional work can also be done to improve the performance of the optimal model. There are sure to be more features about a person that can assist with predicting diabetes, and testing these features will invovle the model to be modified or even completely redone. Using other datasets with both similar and different features would help with further testing whether what was found as the optimal model can still be said for other datasets. These datasets should also attempt to be more balanced in terms of non-diabetes vs. diabetes. Techniques such as ensemble methods could also improve the accuracy of future classification models. 

#### Outline of project

- [Link to Diabetes_Detection](https://github.com/aftseng/Capstone_Project_24.1/blob/main/Diabetes_Detection.ipynb)
- [Link to Capstone_Evaluation](https://github.com/aftseng/Capstone_Project_24.1/blob/main/Capstone_Evaluation.ipynb)
- [Link to Dataset](https://github.com/aftseng/Capstone_Project_24.1/blob/main/diabetes_prediction_dataset.csv)

##### Contact and Further Information

Andrew Tseng

Email: atytseng@gmail.com
