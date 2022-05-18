import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv('coronaryheart.csv') 
print(df.head())
print(df.info())

#Data preprocessing
print(df.isnull().sum())

# percentage of missing data per category
total = df.isnull().sum().sort_values(ascending=False)
percent_total = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])
missing_data = missing[missing['Total']>0]
print(missing_data)

plt.figure(figsize=(9,6))
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data = missing_data)
plt.title('Percentage of missing data by feature')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.show()

print(df.isnull().sum().sum())
df=df.dropna()
print(df.isnull().sum().sum())
print(df.shape)

#Outliers
cols =['age','BMI','heartRate','sysBP','totChol','diaBP']
plt.title("OUTLIERS VISUALIZATION")
for i in cols:
    df[i]
    sns.distplot(df[i],color='grey')
    plt.show()

#Data Visualization(EDA)
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),linewidths=0.1,annot=True)
plt.show()

#sysBP and diaBP,currentSmoker and cigsPerDay are highly correlated with values around 0.8
#sysBP and diaBP and prevalentHyp, diabetes and glucose are correlated to some extent with values arouund 0.62
sns.boxplot(y='age',x='TenYearCHD',data=df)

#Patients who got CHD are in the age group:50- 65
#Patients around the age group:35- 45 does not suffer from CHD mostly
plt.figure()
sns.violinplot(y='cigsPerDay',x='TenYearCHD',data=df)

#It's weird that patients who didn't smoke suffered from CHD
#More the cigarretes they smoke higher chance of getting CHD
plt.figure()
sns.violinplot(y='sysBP',x='TenYearCHD',data=df)

#If your heart rate is in range of 70-80 is safe, but if their heart rate goes above or below can cause CHD
plt.figure()
sns.countplot(x=df['male'], hue=df['TenYearCHD'])

#Males are at higher risk of getting CHD
plt.figure()
sns.countplot(x='currentSmoker',data=df,hue='TenYearCHD')
plt.figure()
sns.countplot(x='prevalentHyp',data=df,hue='TenYearCHD')

#Higher percentage of people having hypertension suffer from CHD
plt.figure()
sns.countplot(x='BPMeds',data=df,hue='TenYearCHD')

#It seems as if 50-60% of patients taking BP meds get CHD
plt.figure()
sns.countplot(x='diabetes',data=df,hue='TenYearCHD')

#It seems as if 60-80% of diabetic patients get CHD
plt.figure()
sns.countplot(x='prevalentStroke',data=df,hue='TenYearCHD')

#In diabetic patients those having higher level of glucose ranging from 200-400, have higher risk of getting CHD".
# plot histogram to see the distribution of the data
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)
plt.show()

#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = df.iloc[:,0:14]  
y = df.iloc[:,-1]    

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  

print(featureScores.nlargest(11,'Score'))  

featureScores = featureScores.sort_values(by='Score', ascending=False)
print(featureScores)

# selecting the 10 most impactful features for the target variable
features_list = featureScores["Specs"].tolist()[:10]
print(features_list)

#new dataframe with selected features
df = df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]
print(df)

df_clean = df
scaler = MinMaxScaler(feature_range=(0,1)) 
scaled_df= pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
scaled_df.describe()
df.describe()

y = scaled_df['TenYearCHD']
X = scaled_df.drop(['TenYearCHD'], axis = 1)

#Split data into train 60% and test 40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=29)

#Resampling imbalanced Dataset
target_count = scaled_df.TenYearCHD.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

sns.countplot(scaled_df.TenYearCHD, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=11)
plt.ylabel('Patient Count',fontsize=11)
plt.title('Count Outcome Heart Disease\n')
#plt.savefig('Balance Heart Disease.png')
plt.show()

#Undersampling methods
shuffled_df = scaled_df.sample(frac=1,random_state=4)
CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 1]
non_CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 0].sample(n=611,random_state=42)
normalized_df = pd.concat([CHD_df, non_CHD_df])
normalized_df.TenYearCHD.value_counts()
sns.countplot(normalized_df.TenYearCHD, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=11)
plt.ylabel('Patient Count',fontsize=11)
plt.title('Count Outcome Heart Disease after Resampling\n')
plt.show()

#SVM Classifier
from sklearn.metrics import classification_report
'''Support Vector Machine'''
print('------Super Vector Machine------')
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)    
svm_pred = svclassifier.predict(X_test)  
'''Analysis Report'''
print()
print("------Accuracy------")
from sklearn.metrics import confusion_matrix, accuracy_score
SVM=(accuracy_score(y_test,svm_pred)*100)
print('Support Vector Machine Accuracy is:',SVM,'%')
print()
print("------Classification Report------")
print(classification_report(svm_pred,y_test))
print('\n')
svm_cm = confusion_matrix(y_test,svm_pred)
print(svm_cm)
svm_result = SVM

'''Decision Tree'''
print()
print('------Decision Tree------')
dtc_up = DecisionTreeClassifier()
dtc_up.fit(X_train, y_train)
dtc_pred = dtc_up.predict(X_test)
'''Analysis Report'''
print()
print("------Accuracy------")
from sklearn.metrics import confusion_matrix, accuracy_score
DT=(accuracy_score(y_test,dtc_pred)*100)
print('Decision Tree Accuracy is:',DT,'%')
print()
print("------Classification Report------")
print(classification_report(dtc_pred,y_test))
print('\n')
dtc_cm = confusion_matrix(y_test,dtc_pred)
print(dtc_cm)
dtc_result = DT

#KNN Classifier
'''KNN Classifier'''
print()
print('------KNN Classifier------')
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
'''Analysis Report'''
print()
print("------Accuracy------")
from sklearn.metrics import confusion_matrix, accuracy_score
KNN=(accuracy_score(y_test,knn_pred)*100)
print('KNN Classifier Accuracy is:',KNN,'%')
print()
print("------Classification Report------")
print(classification_report(knn_pred,y_test))
print('\n')
knn_cm = confusion_matrix(y_test,knn_pred)
print(knn_cm)
knn_result = KNN

#Apply Ensemble Method
print("\n")
print("----------Ensemble Classifier----------")
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('dtc', dtc_up), ('svc', svclassifier),('knn',knn)], voting='hard')
model.fit(X_train,y_train)
eny_pred = model.predict(X_test)
ensemble= accuracy_score(y_test,eny_pred)
cma = confusion_matrix(y_test, eny_pred)
print('\n',cma)
print(classification_report(y_test, eny_pred)) 
print("Ensemble Accuracy(Average):",accuracy_score(y_test,eny_pred)*100)
en_result = accuracy_score(y_test,eny_pred)*100

#Artificial Neural Network (ANN)
"ANN"
from keras.models import Sequential
from keras.layers import Dense
print()
print('------ANN Classifier------')
print()
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, epochs = 5,verbose = 1)

ay_pred = classifier.predict(X_test)
ay_pred = (ay_pred > 0.5)
acm = confusion_matrix(y_test,ay_pred)
print("\n\n\tConfusion Matrix")
print("\t------------------")
print(acm)
"Analysis Report"  
print()
print("------Classification Report------")      
print(classification_report(ay_pred,y_test))
print()
print("------Accuracy------")
print(f"The Accuracy Score :{(accuracy_score(ay_pred,y_test)*100)}")
print()
ann_result = (accuracy_score(ay_pred,y_test)*100)

#Comparison Graph
#Accuracy comparision of Machine learning algorithms
vals=[svm_result,dtc_result,knn_result,en_result,ann_result]
inds=range(len(vals))
labels=["SVM","DTC ","KNN","Ensemble","ANN"]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title("PERFORMANCE COMPARISON")
plt.xlabel("ALGORITHM")
plt.ylabel("ACCURACY")
plt.show()
