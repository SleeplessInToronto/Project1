import numpy as np
from sklearn import preprocessing, cross_validation, svm, neighbors
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

dfvis = df.copy()

df.drop(['customerID'],1, inplace = True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

df.drop(['gender', 'Partner', 'Dependents', 'tenure','TotalCharges', 'PaymentMethod',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 
       'PaperlessBilling'],1, inplace= True)

X= np.array(df.drop(['Churn'],1))

    

y = np.array(df['Churn'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#clf = LinearRegression()
clf=svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[0,1,30]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)

dfvis['churnrate'] = df['Churn']
sns.catplot('Churn', col = 'SeniorCitizen', kind = 'count', data = dfvis)
sns.catplot('Churn', col = 'gender', kind = 'count', data = dfvis)
sns.catplot(x= 'Contract', y= 'churnrate',  data = dfvis, kind = 'bar')
