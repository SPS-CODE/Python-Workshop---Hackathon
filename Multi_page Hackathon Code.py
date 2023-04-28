import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import time
from sklearn.linear_model import LogisticRegression 


st.set_page_config(layout="wide")

df = pd.DataFrame(px.data.gapminder())

progress_bar = st.progress(0)
for i in range(100):
        progress_bar.progress(i+1)
        time.sleep(0.01)

def W2():


    # loading the diabetes dataset to a pandas DataFrame
    diabetes_dataset = pd.read_csv('C:/Users/Abhishek Kumar Yadav/Downloads/diabetes.csv')
    # printing the first 5 rows of the dataset
    diabetes_dataset.head()
    # number of rows and Columns in this dataset
    diabetes_dataset.shape
    # getting the statistical measures of the data
    diabetes_dataset.describe()
    diabetes_dataset['Outcome'].value_counts()
    diabetes_dataset.groupby('Outcome').mean()
    # separating the data and labels
    X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    print(X)
    print(Y)
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    print(standardized_data)
    X = standardized_data
    Y = diabetes_dataset['Outcome']
    print(X)
    print(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    print(X.shape, X_train.shape, X_test.shape)
    classifier = svm.SVC(kernel='linear')
    #training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)
    # accuracy score on the training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score of the training data : ', training_data_accuracy)
    # accuracy score on the test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score of the test data : ', test_data_accuracy)

    st.title('Diabetes Identifier')

    Pregnancies=st.number_input('Enter the Number of pregnancies', min_value=0., max_value=20., step=1.,format="%.2f")
    Glucose=st.number_input ('Enter the Plasma glucose concentration', min_value=60., max_value=200., value=115., step=1.,format="%.2f")
    BloodPressure=st.number_input ('Enter Diastolic blood pressure in mm Hg', min_value=40., max_value=140., value=90., step=1.,format="%.2f")
    SkinThickness=st.number_input ('Enter Triceps skin fold thickness mm', min_value=1., max_value=35., value=15., step=1.,format="%.2f")
    Insulin=st.number_input ('Enter insulin level in mu U/ml', min_value=0., max_value=80., value=40., step=1.,format="%.2f")
    BMI=st.number_input ('Enter Body mass index weight in kg/height in m^2', min_value=15., max_value=180., value=90., step=1.,format="%.2f")
    DiabetesPedigreeFunction=st.number_input ('Enter Diabetes pedigree function', min_value=0., max_value=10., value=5., step=1.,format="%.2f")
    Age=st.number_input ('Enter age in years', min_value=0., max_value=110., value=20., step=1.,format="%.2f")

    input_data=(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    std_data = scaler.transform(input_data_reshaped)
    decision = classifier.predict(std_data)

    if st.button(label='Press to show Results') == True:
        st.write(decision)
    
    if (decision [0]== 0):
        st.write('You are healthy!')
    else:
        st.write('You may have diabetes, consult a doctor!')

def W1():
        heart_data = pd.read_csv('C:/Users/Abhishek Kumar Yadav/Downloads/heart.csv')


        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']

        print (X)
        print (Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        print(X.shape, X_train.shape, X_test.shape)

        model = LogisticRegression()
        model.fit(X_train, Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        print('Accuracy on Training data : ', training_data_accuracy)
        X_test_prediction = model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        print('Accuracy on Test data : ', test_data_accuracy)

        st.title ('Heart Disease Identifier')
        age = st.number_input("Enter age", value=50)
        sex = st.number_input("Enter sex (0 for female, 1 for male)", min_value=0., max_value=1., step=1.,format="%.2f")
        cp = st.number_input("Enter chest pain type (0-3)", min_value=0., max_value=3., value=0., step=1.,format="%.2f")
        trestbps = st.number_input("Enter resting blood pressure in mmHg", min_value=80., max_value=240., value=140., step=1.,format="%.2f")
        chol = st.number_input("Enter serum cholestoral in mg/dl", min_value=60., max_value=280., value=140., step=1.,format="%.2f")
        fbs = st.number_input("Enter fasting blood sugar (0 for no, 1 for yes)", min_value=0., max_value=1., step=1.,format="%.2f")
        restecg = st.number_input("Enter resting electrocardiographic results (0-2)", min_value=0., max_value=2., step=1.,format="%.2f")
        thalach = st.number_input("Enter maximum heart rate achieved", min_value=60., max_value=200., value=160., step=1.,format="%.2f")
        exang = st.number_input("Enter exercise induced angina (0 for no, 1 for yes)", min_value=0., max_value=1., step=1.,format="%.2f")
        oldpeak = st.number_input("Enter ST depression induced by exercise relative to rest", value=0., step=1.,format="%.2f")
        slope = st.number_input("Enter the slope of the peak exercise ST segment (0-2)", min_value=0., max_value=2., step=1.,format="%.2f")
        ca = st.number_input("Enter number of major vessels (0-3) colored by flourosopy", min_value=0., max_value=3., step=1.,format="%.2f")
        thal = st.number_input("Enter type of thalassemia 1 = Normal; 2 = Fixed defect; 3 = Reversible defect", min_value=0., max_value=3., step=1.,format="%.2f")
        
        input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        input_data_as_numpy_array = np.asarray(input_data) 


        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)

        if st.button(label='Click to show Results'):
            st.write (prediction)
        
        if (prediction [0]==0):
            st.write('You are healthy!')
        else:  
            st.write('You may have a heart disease, consult a Doctor!')

st.header("Heart Disease and Diabetes Identifier")
page = st.sidebar.selectbox('Select page',['W2','W1']) 
if page == 'W2':
     W2()
else: 
    W1()
