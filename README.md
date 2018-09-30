# HR-Churn-Modelling
In this project I have tried to predict if an employee is churning by setting up extensive machine learning pipeline.

## Contents:--  
##### 1) Problem Description  
##### 2) DataSet Description  
##### 3) Target Description   
##### 4) Classification Metrics  
##### 5) Engineering Features  
##### 6) Machine Learning Pipeline Setup  
###### . Scaling data with standardization  
###### . Baseline model  
###### . comparing different models  
###### . Voting Classifiers  
##### 7) XGBoost  
##### 8) Feature Selection  
##### 9) Feature Extraction  
##### 10) Bayesian Optimization   
##### 11) Final Tunned Xgboost Model.  


### Probelm Description:---  
Determine what factors predict an employee leaving his/her job. For this project you will be evaluating a dataset composed of human resources data. Your objective is to build a model (or models) that predict whether or not an employee is likely to leave his/her job based on characteristics in the dataset.  

### Dataset Description:---  
**The dataset contains around 15000 rows with 9 independent features and 1 dependent variable target which determines wheather an employee will leave or not.** 
<class 'pandas.core.frame.DataFrame'>  
RangeIndex: 14999 entries, 0 to 14998  
Data columns (total 10 columns):  
satisfaction -----            14999 non-null float64  
evaluation  -----            14999 non-null float64  
number_of_projects-----      14999 non-null int64  
average_montly_hours  -----  14999 non-null int64  
time_spend_company   ----   14999 non-null int64  
work_accident   -----        14999 non-null int64  
churn          -----         14999 non-null int64  
promotion        ------       14999 non-null int64  
department  ------           14999 non-null object  
salary      ------            14999 non-null object  
dtypes: float64(2), int64(6), object(2)  
memory usage: 1.1+ MB  

### Target Description:--  
The target distribution is highly imbalanced and this type of problems are often imbalanced.  
**Barplot showing the count of different categories**    
![alt target_Description](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/target_desp.png)  
As it could be observed eaisy that it is binary classification task.   

### Classification Metrics:---  
we have a set of differernt classification metrics but the most suitable one is f1_score as it is harmonic mean of recall   
ans precision.  
We could have even used ROC_AUC score as the problem is imbalanced but I found f1_score more suitable as of problem description.  

### Engineering Features:---  
We have created feautres bascially by two method   
1)Transformation:--   
2)Aggregation:---    using mean,min,max,sum,std we have created whole lot of features.  
3)Some of the features are created as a result of Exploratory Data Analysis.  

### Machine Learning Pipeline:--  
#### BaseLine Model:--  
The best model to start with is random forest as it supports both classification and regression task.   
Description of the baseline model  
![alt target_Description](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/baseline_model.png)  

#### Comparing Different Model:-    
we have started with some of the best models like Logistic Regression and KNN.  
This graph describes the performance of the models:--  
![alt model_comparasion](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/comparing_models.png)  
As it can be seen that extra tree is performing well along with MLPclassifier.  

#### Voting Classifier:--  
we went on making a hard voting classifier using Extratrees,randomforest and MLPclassifier.  
It did performed failrly well.  

#### XGBoost :--  
We then moved to use our advanced model using xgboost which was untunned and the feature selected by xgboost are as:--  
![alt xgboost_feautre_imp](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/xgboost_feature_imp.png)  


we then used cross_validation using xgboost and compared the mean error rate in train and test sets:-- 
![alt error](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/error.png)  

### Feature Selection:--  
we had a considerably bigger data set then we started with .  
at this stage we have 202 features created by feature engineering step.  
we then removed all those features having correlation more then 95% among themseleves and then used recursive feature   
elimination to select only best performing features.  
This reduced our dataset to 114 feaures.  

### Feature Extraction:--  
we used pca,ica,umap to represent the dataset in to three dimensional space and we used these features in the mmodelling.  
for umap:--  
![alt umap](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/umap.png)  
for pca:--  
![alt pca](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/pca.png)  
for ica:--  
![alt ica](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/ica.png)  

### Bayesian Optimization:--  
we have different techniques for selecting the best parameters for the modelling algorithm but they are not efficient as  
of bayesian optimization.  
Feature space:--    
space = {  
    'boosting_type': hp.choice('boosting_type',    
                              [{'boosting_type': 'gbdt',
                                'subsample': hp.uniform('gdbt_subsample', 0.5, 1),
                                'subsample_freq': hp.quniform('gbdt_subsample_freq', 1, 10, 1)}, 
                               {'boosting_type': 'dart', 
                                 'subsample': hp.uniform('dart_subsample', 0.5, 1),
                                 'subsample_freq': hp.quniform('dart_subsample_freq', 1, 10, 1),
                                 'drop_rate': hp.uniform('dart_drop_rate', 0.1, 0.5)},
                                {'boosting_type': 'goss',
                                 'subsample': 1.0,
                                 'subsample_freq': 0}]),  
    'limit_max_depth': hp.choice('limit_max_depth', [True, False]),  
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),  
    'num_leaves': hp.quniform('num_leaves', 3, 50, 1),  
    'learning_rate': hp.loguniform('learning_rate',   
                                   np.log(0.025),   
                                   np.log(0.25)),  
    'subsample_for_bin': hp.quniform('subsample_for_bin', 2000, 100000, 2000),  
    'min_child_samples': hp.quniform('min_child_samples', 5, 80, 5),  
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),  
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),  
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),  
    'objective':'binary:logistic'  
}  

**Best Parameters**  
{'boosting_type': 1,  
 'colsample_by_tree': 0.9705787113662703,  
 'dart_drop_rate': 0.10481709074419068,  
 'dart_subsample': 0.8807619552902802,  
 'dart_subsample_freq': 10.0,  
 'learning_rate': 0.2488211061090212,  
 'limit_max_depth': 1,  
 'max_depth': 11,  
 'min_child_samples': 10.0,  
 'num_leaves': 18.0,  
 'reg_alpha': 0.1236107627491404,  
 'reg_lambda': 0.5912175686153809,  
 'subsample_for_bin': 30000.0}  
 
 ### Final Modelling:--  
 Using the parameters obtained by the optimization we have reachecd the f1_score of **98.79%**.  
 This confusion matrix better describes it.  
 ![alt final](https://github.com/Ankit01Mishra/HR-Churn-Modelling/blob/master/result_images/final.png)   
 
 **Other Scores**:--  
 roc_auc_score: ---  99.57%  
 precision: -- 97.75%  
 recall: -- 99.84%  
 accuracy_score: -- 99.34%  
 
 
 ## Things to try:--  
 we haven't tried deep learning and advanced stacking of the models.  
 This can lead to increase the performance of the model.  
 
 
