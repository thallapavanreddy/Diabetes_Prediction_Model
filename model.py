import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import minmax_scale

#from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

data = pd.read_csv('diabetes.csv')

data = data[data['SkinThickness']<70]
y = data.iloc[:,-1]
x = data.iloc[:,:-1]

for col in x.columns:
    if col != 'Pregnancies':
        x[col] = x[col].replace({0:np.nan})

missing = pd.DataFrame({
    'name': x.columns,
    'count': x.isnull().sum(),
    'percentage': x.isnull().sum()/x.shape[0]*100
})

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
for col in X_train.columns:
    X_train[col] = X_train[col].fillna(X_train[col].mean()).astype('float64')
    X_test[col] = X_test[col].fillna(X_train[col].mean()).astype('float64')

Logi_Reg_Model = LogisticRegression()
Logi_Reg_Model.fit(X_train, y_train)

y_predict = Logi_Reg_Model.predict(X_test)
y_predict_pa = Logi_Reg_Model.predict_proba(X_test)[::,1]


confusionMatrix = confusion_matrix(y_test, y_predict)
print(confusionMatrix)

print('F1_Score : {}'.format(f1_score(y_test, y_predict)))

auc = round(roc_auc_score(y_test, y_predict_pa), ndigits=8)
fpr, tpr, thrshould = roc_curve(y_test, y_predict_pa)

import joblib
model_Diabetes = 'diabetesML.pkl'
with open(model_Diabetes, 'wb') as file:
    joblib.dump(Logi_Reg_Model, file)

#model = joblib.load(filename='diabetesML.pkl')
#model.predict(X_test)

plt.figure(figsize=(12,12))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under The Curve AUC-ROC')

plt.legend(loc= 7)
plt.show()



