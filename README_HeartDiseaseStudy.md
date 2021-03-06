# HeartDiseaseStudy

DataSet: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

This program selects features from the Cleveland heart disease study on
the UC Irvine Machine Learning Repository.  It does this by maximizing
the accuracy of a K-Nearest Neighbors classifier.  The initial features
are the following:
 1. age: continuous
 2. sex: categorical, 2 values {0: female, 1: male}
 3. cp (chest pain type): categorical, 4 values
    {1: typical angina, 2: atypical angina, 3: non-angina,
     4: asymptomatic angina}
 4. restbp (resting blood pressure on admission to hospital): continuous (mmHg)
 5. chol (serum cholesterol level): continuous (mg/dl)
 6. fbs (fasting blood sugar): categorical, 2 values
    {0: <= 120 mg/dl, 1: > 120 mg/dl}
 7. restecg (resting electrocardiography): categorical, 3 values
    {0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy}
 8. thalach (maximum heart rate achieved): continuous
 9. exang (exercise induced angina): categorical, 2 values {0: no, 1: yes}
10. oldpeak (ST depression induced by exercise relative to rest): continuous
11. slope (slope of peak exercise ST segment): categorical, 3 values
    {1: upsloping, 2: flat, 3: downsloping}
12. ca (number of major vessels colored by fluoroscopy): discrete (0,1,2,3)
13. thal: categorical, 3 values {3: normal, 6: fixed defect,
    7: reversible defect}
14. num (diagnosis of heart disease): categorical, 5 values
    {0: less than 50% narrowing in any major vessel, 1-4: more than
    50% narrowing in 1-4 vessels}
    
The actual number of feature variables (after converting categorical variables
to dummy ones) is:
1 (age) + 1 (sex) + 4 (cp) + 1 (restbp) + 1 (chol) + 1 (fbs) + 3 (restecg) +
1 (thalach) + 1 (exang) + 1 (oldpeak) + 3 (slope) + 1 (ca) + 3 (thal) = 22

# What are we going to do?

We are going to test different ML models one by one on different parameteres and check their accuracies. As the classes to classify are too much compared to the amount of samples we have, we might not good very good accuracies.
