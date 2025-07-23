import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data={
    'math':[78,65,78,45,30,60,80,40,20,50],
    'science':[20,68,45,82,63,81,20,40,60,90],
    'english':[30,70,45,65,98,42,30,30,60,60],
    'result':['fail','pass','pass','pass','fail','fail','fail','fail','fail','pass']
    }
df=pd.DataFrame(data)
print(df)
df['result']=df['result'].map({'pass':1,'fail':0})
x=df[['math','science','english']]
y=df['result']
x_train,X_test,y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=40)
print(X_test)
model=LogisticRegression()
model.fit(x_train,y_train)

res=model.predict(X_test)
print("Accuracy",accuracy_score(Y_test,res))
new_student=pd.DataFrame([[60,70,80]],columns=['math','science','english'])
predict=model.predict(new_student)
print(predict[0])
joblib.dump(model,'model.pkl')





