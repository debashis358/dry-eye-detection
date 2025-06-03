import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,classification_report,roc_curve,confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

df=pd.read_csv('Dry_Eye_Dataset.csv')

df.head()

df.info()

df.duplicated().sum()

# Missing Values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing Value Percentage:")
print(missing_percentage)

df.describe()

for i in df.select_dtypes(include='object').columns:
    print(f'{i}\n',df[i].unique())

# we will Split 'Blood pressure' into 'Systolic' and 'Diastolic'
df[['Systolic BP', 'Diastolic BP']] = df['Blood pressure'].str.split('/', expand=True)

# Convert the new columns to numeric
df['Systolic BP'] = pd.to_numeric(df['Systolic BP'], errors='coerce')
df['Diastolic BP'] = pd.to_numeric(df['Diastolic BP'], errors='coerce')

# Drop the original 'Blood pressure' column
df = df.drop(columns=['Blood pressure'])

# checking for outliers
numeric=df.select_dtypes(include=np.number).columns
plt.figure(figsize=(12,18))
t=1
for i in numeric:
    plt.subplot(6,2,t)
    sns.boxplot(df[i])
    plt.title(f'Boxplot of {i}')
    t+=1
plt.tight_layout()
plt.show()

numeric=df.select_dtypes(include=np.number).columns
plt.figure(figsize=(12,18))
t=1
for i in numeric:
    plt.subplot(7,2,t)
    sns.histplot(df[i])
    plt.title(f'Distribution of {i}')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    t+=1
plt.tight_layout()
plt.show()

for i in numeric:
    print(f'Skewness of {i} :',df[i].skew())

categoric=df.select_dtypes(include='object').columns
for i in categoric:
    print(df[i].value_counts(normalize=True)*100)



plt.figure(figsize=(12, 18))
t = 1

for i in categoric:
    plt.subplot(10, 2, t)
    df[i].value_counts(normalize=True).plot(kind='bar')
    plt.title(f'Distribution of {i}')
    plt.ylabel('')
    t += 1

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 18))
t=1
for i in numeric:
    plt.subplot(7,2,t)
    sns.boxplot(y=df[i], x=df['Dry Eye Disease'])
    plt.title(f'{i} vs Dry Eye Disease')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    t+=1
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 24))
t = 1

for col in categoric:
    if col != 'Dry Eye Disease':
        ax = plt.subplot(8, 2, t)
        pd.crosstab(df[col], df['Dry Eye Disease'], normalize='index').plot(
            kind='bar', ax=ax, legend=False)
        ax.set_title(f'{col} vs Dry Eye Disease')
        ax.set_ylabel('Proportion')
        ax.set_xlabel(col)
        t += 1

plt.tight_layout()
plt.legend(['No', 'Yes'], title='Dry Eye Disease', bbox_to_anchor=(1.05, 4), loc='upper left')
plt.show()

df['Pulse_Pressure']=df['Systolic BP']-df['Diastolic BP']

sns.boxplot(x=df['Dry Eye Disease'],y=df['Pulse_Pressure'])
plt.show()

# adding one more feature 'BMI'

df['BMI']=df['Weight'] / (df['Height']/100)**2

sns.boxplot(x=df['Dry Eye Disease'],y=df['BMI'])
plt.show()

# Adding Blood pressure category based on systolic and diastolic metrics using data/facts from American Heart Association

def classify_bp(row):
    systolic = row['Systolic BP']
    diastolic = row['Diastolic BP']
    
    if systolic > 180 or diastolic > 120:
        return 'Hypertensive Crisis'
    elif systolic >= 140 or diastolic >= 90:
        return 'Hypertension Stage 2'
    elif systolic >= 130 or diastolic >= 80:
        return 'Hypertension Stage 1'
    elif systolic >= 120 and diastolic < 80:
        return 'Elevated'
    else:
        return 'Normal'
# Create new column
df['BP_category'] = df.apply(classify_bp, axis=1)

pd.crosstab(df['BP_category'], df['Dry Eye Disease'], normalize='index').plot(kind='bar',legend=False)
plt.title('BP_category vs Dry eye disease')

# Categorizing sleep duration based on data/facts provided by WHO

def categorize_sleep(duration):
    if duration < 7:
        return 'Short'
    elif 7 <= duration <= 9:
        return 'Healthy'
    else:
        return 'Long'

df['Sleep_category'] = df['Sleep duration'].apply(categorize_sleep)

pd.crosstab(df['Sleep_category'], df['Dry Eye Disease'], normalize='index').plot(kind='bar',legend=False)

#categorizing average screen time using data/facts from National Institute of health

def categorize_screen_time(hours):
    if hours <= 2:
        return 'Low'
    elif hours <= 6:
        return 'Moderate'
    elif hours <= 9:
        return 'High'
    else:
        return 'Very High'

df['Screen_Time_Category'] = df['Average screen time'].apply(categorize_screen_time)

pd.crosstab(df['Screen_Time_Category'], df['Dry Eye Disease'], normalize='index').plot(kind='bar',legend=False)

# Using heat map checking corelation after adding new features

numeric=df.select_dtypes(include=np.number).columns
plt.figure(figsize=(15,15))
sns.heatmap(df[numeric].corr(),annot=True,cmap='viridis')

sns.pairplot(df)
plt.show()

df.head()

df.info()

categoric=df.select_dtypes(include='object').columns

from scipy.stats import chi2_contingency

for col in categoric:
    if col !='Dry Eye Disease':
        contingency_table = pd.crosstab(df[col], df['Dry Eye Disease'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test for {col}:")
        print(f"  p-value: {p}")
        if p < 0.05:
            print("  Conclusion: There is a statistically significant association between", col, "and Dry Eye Disease.")
        else:
            print("  Conclusion: There is no statistically significant association between", col, "and Dry Eye Disease.")

from scipy.stats import ttest_ind
numeric=df.select_dtypes(include=np.number).columns
for i in numeric:
    group1=df[df['Dry Eye Disease']=='Y'][i]
    group2=df[df['Dry Eye Disease']=='N'][i]
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    print(f"ttest_ind for {i}:")
    print(f"  p-value: {p_val}")
    if p_val < 0.05:
        print("  Conclusion: There is a statistically significant association between", i, "and Dry Eye Disease.")
    else:
        print("  Conclusion: There is no statistically significant association between", i, "and Dry Eye Disease.")



categoric=df.select_dtypes(include='object').columns
for col in categoric:
    if col !='Discomfort Eye-strain':
        contingency_table = pd.crosstab(df[col], df['Discomfort Eye-strain'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test for {col}:")
        print(f"  p-value: {p}")
        if p < 0.05:
            print("  Conclusion: There is a statistically significant association between", col, "and Discomfort_Eye_strain.")
        else:
            print("  Conclusion: There is no statistically significant association between", col, "and Discomfort_Eye_strain.")

numeric=df.select_dtypes(include=np.number).columns
for i in numeric:
    group1=df[df['Discomfort Eye-strain']=='Y'][i]
    group2=df[df['Discomfort Eye-strain']=='N'][i]
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    print(f"ttest_ind for {i}:")
    print(f"  p-value: {p_val}")
    if p_val < 0.05:
        print("  Conclusion: There is a statistically significant association between", i, "and Discomfort_Eye_strain.")
    else:
        print("  Conclusion: There is no statistically significant association between", i, "and Discomfort_Eye_strain.")

for col in categoric:
    if col !='Redness in eye':
        contingency_table = pd.crosstab(df[col], df['Redness in eye'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test for {col}:")
        print(f"  p-value: {p}")
        if p < 0.05:
            print("  Conclusion: There is a statistically significant association between", col, "and Redness_in_eye.")
        else:
            print("  Conclusion: There is no statistically significant association between", col, "and Redness_in_eye.")

for i in numeric:
    group1=df[df['Redness in eye']=='Y'][i]
    group2=df[df['Redness in eye']=='N'][i]
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    print(f"ttest_ind for {i}:")
    print(f"  p-value: {p_val}")
    if p_val < 0.05:
        print("  Conclusion: There is a statistically significant association between", i, "and Redness_in_eye.")
    else:
        print("  Conclusion: There is no statistically significant association between", i, "and Redness_in_eye.")

for col in categoric:
    if col !='Itchiness/Irritation in eye':
        contingency_table = pd.crosstab(df[col], df['Itchiness/Irritation in eye'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test for {col}:")
        print(f"  p-value: {p}")
        if p < 0.05:
            print("  Conclusion: There is a statistically significant association between", col, "and Itchiness_Irritation_in_eye.")
        else:
            print("  Conclusion: There is no statistically significant association between", col, "and Itchiness_Irritation_in_eye.")

for i in numeric:
    group1=df[df['Redness in eye']=='Y'][i]
    group2=df[df['Redness in eye']=='N'][i]
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    print(f"ttest_ind for {i}:")
    print(f"  p-value: {p_val}")
    if p_val < 0.05:
        print("  Conclusion: There is a statistically significant association between", i, "and Redness_in_eye.")
    else:
        print("  Conclusion: There is no statistically significant association between", i, "and Redness_in_eye.")

df.columns = df.columns.str.strip()  # remove leading/trailing spaces
df.columns = df.columns.str.replace('[^0-9a-zA-Z]+', '_', regex=True)

df.columns

categoric=df.select_dtypes(include='object').columns
categoric

#df=df.drop('symptom_severity',axis=1)

df1=df.copy()

df1=df1.drop(['Height','Weight'],axis=1)

cols=['Sleep_disorder', 'Wake_up_during_night',
       'Feel_sleepy_during_day', 'Caffeine_consumption', 'Alcohol_consumption',
       'Smoking', 'Medical_issue', 'Ongoing_medication',
       'Smart_device_before_bed', 'Blue_light_filter', 'Discomfort_Eye_strain',
       'Redness_in_eye', 'Itchiness_Irritation_in_eye', 'Dry_Eye_Disease']

df1[cols] = df1[cols].applymap(lambda x: 1 if x == 'Y' else 0)

df1 = pd.get_dummies(df1, columns=['BP_category', 'Sleep_category', 'Screen_Time_Category'], drop_first=True,dtype=int)

df1['Gender']=df1['Gender'].apply(lambda x:1 if x=='M' else 0)

numcol=['Age', 'Sleep_duration', 'Sleep_quality', 'Stress_level', 'Heart_rate','Daily_steps', 'Physical_activity',
        'Average_screen_time', 'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure','BMI']

ss=StandardScaler()
df1[numcol]=ss.fit_transform(df1[numcol])

df1.head(5)

df1.info()

#df1.to_csv('encoded_dry_eye.csv', index=False)

# train test split

X=df1.drop(['Dry_Eye_Disease'],axis=1)
y=df1['Dry_Eye_Disease']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

print('Xtrain',X_train.shape)
print('Xtest',X_test.shape)
print('ytrain',y_train.shape)
print('ytest',y_test.shape)

print('y_train')
y_train.value_counts()

print('y_test')
y_test.value_counts()

def metrics(y_test,y_pred,model):
    print(model)
    print('accuracy',accuracy_score(y_test,y_pred))
    print('precision',precision_score(y_test,y_pred))
    print('recall',recall_score(y_test,y_pred))
    print('fi score',f1_score(y_test,y_pred))
    print('classification report',classification_report(y_test,y_pred))

def plot_roc_curve(y_true, y_probs, model):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Calculate AUC Score
    auc_score = roc_auc_score(y_true, y_probs)
    print(f'ROC-AUC Score for {model}: {auc_score:.2f}')

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', label=f'{model} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model}')
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_check(X,numcol):
    vif=pd.DataFrame()
    vif['feature']=X[numcol].columns
    vif['vifscore']=[variance_inflation_factor(X[numcol].values,i) for i in range(X[numcol].shape[1])]
    vif['vifscore']=round(vif['vifscore'],2)
    vif=vif.sort_values(by='vifscore',ascending=False)
    return vif

def imp_feature(model):
    importance = model.feature_importances_
    features = X_train.columns
    
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10,6))
    plt.barh(feat_df['Feature'], feat_df['Importance'])
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

lr=LogisticRegression()
lr_model=lr.fit(X_train,y_train)


y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

metrics(y_train, y_train_pred,'train_metrics')

metrics(y_test, y_test_pred,'test_metrics')

lr=LogisticRegression()
lr_model=lr.fit(X_train,y_train)
y_pred=lr_model.predict(X_test)
y_pred_proba=lr_model.predict_proba(X_test)[:,1]

metrics(y_test,y_pred,'Logistic Regression')

plot_confusion_matrix(y_test,y_pred)

metrics(y_test,y_pred,'Logistic Regression')

vif_check(X_train,numcol)

X=df1.drop(['Dry_Eye_Disease','Diastolic_BP'],axis=1)
y=df1['Dry_Eye_Disease']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print('Xtrain',X_train.shape)
print('Xtest',X_test.shape)
print('ytrain',y_train.shape)
print('ytest',y_test.shape)

numcol1=['Age', 'Sleep_duration', 'Sleep_quality', 'Stress_level', 'Heart_rate','Daily_steps', 'Physical_activity',
        'Average_screen_time', 'Systolic_BP', 'Pulse_Pressure','BMI']
ss=StandardScaler()
X_train[numcol1]=ss.fit_transform(X_train[numcol1])
X_test[numcol1]=ss.transform(X_test[numcol1])

vif_check(X_train,numcol1)

lr=LogisticRegression()
lr_model=lr.fit(X_train,y_train)
train_pred=lr_model.predict(X_train)
y_pred=lr_model.predict(X_test)
y_pred_proba=lr_model.predict_proba(X_test)[:,-1]
y_pred_cust = (y_pred_proba >= 0.4).astype(int)

metrics(y_train,train_pred,'train accuracy')

metrics(y_test,y_pred,'test accuracy')

plot_confusion_matrix(y_test,y_pred)

plot_roc_curve(y_test,y_pred_proba,'lr-model')

metrics(y_test,y_pred_cust,'threshold 0.4')

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
train_pred=dt_model.predict(X_train)
y_pred = dt_model.predict(X_test)
y_pred_prob = dt_model.predict_proba(X_test)[:, 1]

metrics(y_train,train_pred,'train metrics')

metrics(y_test,y_pred,'test metrics')

plot_confusion_matrix(y_test,y_pred)

#Tunned Dt

param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [5, 10, 20, 50],
    'min_samples_leaf': [2, 4, 10]
}

dt_serach=GridSearchCV(estimator=dt_model,param_grid=param,cv=5,n_jobs=-1,verbose=1)
model=dt_serach.fit(X_train,y_train)
print(model.best_params_)

dt_model = DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=2,min_samples_split=5)
dt_model.fit(X_train, y_train)
train_pred=dt_model.predict(X_train)
y_pred = dt_model.predict(X_test)
y_pred_proba=dt_model.predict_proba(X_test)[:,1]

metrics(y_train,train_pred,'train prediction')

metrics(y_test,y_pred,'Test prediction')

plot_confusion_matrix(y_test,y_pred)

plot_roc_curve(y_test,y_pred_proba,'dt-model')

rf = RandomForestClassifier()
rf_model = rf.fit(X_train, y_train)
train_pred=rf_model.predict(X_train)
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:,1]

metrics(y_train, train_pred, 'train_metrics')

metrics(y_test,y_pred,'test metrics')

plot_confusion_matrix(y_test,y_pred)

param = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

random_search = GridSearchCV(
    estimator=rf,
    param_grid=param,
    cv=5,
    verbose=1,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Best Params:", random_search.best_params_)

rf = RandomForestClassifier(class_weight='balanced',n_estimators=200,min_samples_split=5,random_state=42,max_depth=20)
rf_model = rf.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
metrics(y_test, y_pred_rf, 'Random Forest')

plot_confusion_matrix(y_test,y_pred_rf)

imp_feature(rf)

ada=AdaBoostClassifier(random_state=42)
ad_model=ada.fit(X_train,y_train)
y_pred=ad_model.predict(X_test)

metrics(y_test,y_pred,'ad_model')

plot_confusion_matrix(y_test,y_pred)

#tunning ada boost

base_dt=DecisionTreeClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'algorithm': ['SAMME'],
    'estimator': [DecisionTreeClassifier(max_depth=1),
                  DecisionTreeClassifier(max_depth=2),
                  DecisionTreeClassifier(max_depth=3)]
}

ada = AdaBoostClassifier(estimator=base_dt)
grid = GridSearchCV(estimator=ada, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_sel, y_train)
print("Best Params:", grid.best_params_)

ada=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),algorithm='SAMME',learning_rate=0.01,n_estimators=50,random_state=42)
ad_model=ada.fit(X_train,y_train)
y_pred=ad_model.predict(X_test)

metrics(y_test,y_pred,'tunned_model')

imp_feature(ad_model)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

metrics(y_test,y_pred,'GB_model')

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

gb = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,max_features='log2',n_estimators=100,subsample=1.0,random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

metrics(y_test,y_pred,'GB-tunned')

plot_confusion_matrix(y_test,y_pred)

imp_feature(gb)

num_neg = sum(y_train == 0)
num_pos = sum(y_train == 1)

scale_pos_weight = num_pos/ num_neg
print("scale_pos_weight =", scale_pos_weight)

XGB = XGBClassifier()
xgb_model=XGB.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

metrics(y_test,y_pred,'XG-model')

plot_confusion_matrix(y_test,y_pred)

param = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5],
    'scale_pos_weight': [1, 2, 5] 
}

grid_search = GridSearchCV(
    estimator=XGB,
    param_grid=param,
    scoring='recall',
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

XGB = XGBClassifier(learning_rate=0.01,max_depth=3,min_child_weight=1,n_estimators=100,scale_pos_weight=1.81)
xgb_model=XGB.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
y_pred_prob=xgb_model.predict_proba(X_test)[:,1]
y_pred_cust=(y_pred_prob>=0.7).astype(int)

metrics(y_test,y_pred_cust,'tunned model')

svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)


metrics(y_test,y_pred,'SVM_model')

plot_confusion_matrix(y_test,y_pred)

#tunned svm

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

svm = SVC(C=1,gamma='scale',kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

metrics(y_test,y_pred,'tunned_svm')

plot_confusion_matrix(y_test,y_pred)

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

metrics(y_test,y_pred,'LGBM')

plot_confusion_matrix(y_test,y_pred)

#tunning

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 10, -1],
    'num_leaves': [31, 64],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)



import shap
model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary Plot (bar shows global feature importance)
shap.plots.bar(shap_values, max_display=10)

# Models
models = [dt_model,rf, ada, gb, XGB, lgbm, svm]  # your pre-trained models
model_names = ['decision tree', 'random forest', 'Ada Boost', 'GradientBoost', 'XGBoost', 'Light GBM', 'Support Vector']


# Number of features to select
n_features_to_select = 30
for model, name in zip(models, model_names):
    # RFE Feature Selection
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    print(f'{name}\n',selected_features)

#Decision Tree on RFE features


selected_features=['Gender', 'Age', 'Sleep_duration', 'Sleep_quality', 'Stress_level',
       'Heart_rate', 'Daily_steps', 'Physical_activity', 'Sleep_disorder',
       'Wake_up_during_night', 'Feel_sleepy_during_day',
       'Caffeine_consumption', 'Alcohol_consumption', 'Smoking',
       'Medical_issue', 'Average_screen_time', 'Blue_light_filter',
       'Discomfort_Eye_strain', 'Redness_in_eye',
       'Itchiness_Irritation_in_eye', 'Systolic_BP', 'Diastolic_BP',
       'Pulse_Pressure', 'BMI', 'BP_category_Hypertension Stage 1',
       'BP_category_Hypertension Stage 2', 'BP_category_Normal',
       'Sleep_category_Long', 'Sleep_category_Short',
       'Screen_Time_Category_Low']

X_train_sel=X_train[selected_features]
X_test_sel=X_test[selected_features]

dt_rfe = DecisionTreeClassifier()
dt_rfe.fit(X_train_sel, y_train)
train_pred=dt_rfe.predict(X_train_sel)
y_pred = dt_rfe.predict(X_test_sel)
y_pred_prob = dt_rfe.predict_proba(X_test_sel)[:, 1]

metrics(y_train,train_pred,'train metrics')

metrics(y_test,y_pred,'test metrics')

param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [5, 10, 20, 50],
    'min_samples_leaf': [2, 4, 10]
}

dt_serach=GridSearchCV(estimator=dt_rfe,param_grid=param,cv=5,n_jobs=-1,verbose=1)
model=dt_serach.fit(X_train_sel,y_train)
print(model.best_params_)

dt_rfe = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=2,min_samples_split=2,max_depth=4,random_state=42)
dt_rfe.fit(X_train_sel, y_train)
y_pred = dt_rfe.predict(X_test_sel)
metrics(y_test,y_pred,'dt rfe')

# Random Forest

rf_model_rfe = RandomForestClassifier(random_state=42)
rf_model_rfe.fit(X_train_sel, y_train)
y_pred_rf = rf_model_rfe.predict(X_test_sel)
y_pred_prob=rf_model_rfe.predict_proba(X_test_sel)[:,1]
y_pred_cust = (y_pred_prob > 0.35).astype(int)
metrics(y_test, y_pred, 'Random Forest')

#ada boost
ada=AdaBoostClassifier(random_state=42)
ad_rfe=ada.fit(X_train_sel,y_train)
y_pred=ad_rfe.predict(X_test_sel)

metrics(y_test,y_pred,'adaboost rfe')

#Gradient boost

gb_rfe = GradientBoostingClassifier()
gb_rfe.fit(X_train_sel, y_train)
y_pred = gb_rfe.predict(X_test_sel)

metrics(y_test,y_pred,'gradient boost rfe')

# XGBoost

XGB = XGBClassifier()
xgb_rfe=XGB.fit(X_train_sel, y_train)
y_pred = xgb_rfe.predict(X_test_sel)

metrics(y_test,y_pred,'XGB rfe')













metrics(y_test,y_pred,'shap xgb model')

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred_prob=lr.predict_proba(X_test)[:,1]
y_pred_cust = (y_pred_prob >= 0.55).astype(int)

metrics(y_test,y_pred_cust,'l model')



voting_clf = VotingClassifier(estimators=[
    ('dt', dt_model),
    ('rf', rf),
    ('ada', ada),
    ('gb',gb),
    ('xgb', XGB),
    ('lbgm',lgbm),
], voting='soft')  # Use 'hard' for majority vote, 'soft' for probabilities

# Fit VotingClassifier
voting_clf.fit(X_train, y_train)

# Predict
y_pred_voting = voting_clf.predict(X_test)


metrics(y_test,y_pred_voting,'voting')

def metric(y_test, y_pred, model):
    print(model)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, prec, rec, f1

models = [
    ('Logistic Regression', lr_model),
    ('Decision Tree', dt_model),
    ('Random Forest', rf),
    ('Gradient Boosting', gb),
    ('XGBoost', XGB),
    ('LightGBM', lgbm),
    ('AdaBoost', ada),
    ('SVC', svm),
    ('Voting Classifier', voting_clf)
]
results = []

# Loop through models
for name, model in models:
    y_pred = model.predict(X_test)
    acc, prec, rec, f1 = metric(y_test, y_pred, name)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)
df_results

df_results

