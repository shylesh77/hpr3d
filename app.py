import pandas as pd
import scipy.stats as st
import numpy as np
import statsmodels.api as sm
from flask import Flask, render_template,request
from email.message import EmailMessage
import smtplib
import model
import ard
patient={}
sex=0
age=0
currentSmoker=0
cigsPerDay=0
BPMeds=0
prevalentStroke=0
prevalentHyp=0
diabetes=0
totChol=0
sysBP=0
diaBP=0
BMI=0.0
pulseRate=0
glucose=0
m=""
msg=EmailMessage()
msg['Subject']='Heart Disease Diagnosis'
msg['From']='shreeshyleshronaldo@gmail.com'
app = Flask(__name__,static_folder='static')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/heart_control/')
def link():
    return render_template('index.html')
@app.route('/heartRate')
def heartRate():
    global pulseRate
    pulseRate=int(ard.heart_control())
    return render_template('temp.html',heartRate=pulseRate)
@app.route('/predic')
def predic():
    global sex
    global age
    global currentSmoker
    global cigsPerDay
    global BPMeds
    global prevalentHyp
    global prevalentStroke
    global diabetes
    global totChol
    global sysBP
    global diaBP
    global BMI
    global glucose
    global m
    global pulseRate
    train=pd.read_csv('framingham.csv')
    train.head()
    train.isnull().sum()
    train.dropna(axis=0,inplace=True)
    from statsmodels.tools import add_constant as add_constant
    train_constant = add_constant(train)
    train_constant.head()
    st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
    cols=train_constant.columns[:-1]
    model=sm.Logit(train.TenYearCHD,train_constant[cols])
    result=model.fit()
    result.summary()
    import sklearn
    new_features=train[['age','cigsPerDay','totChol','sysBP','BMI','heartRate','glucose','TenYearCHD']]
    x=new_features.iloc[:,:-1]
    y=new_features.iloc[:,-1]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
    from sklearn.linear_model import LogisticRegression
    logreg=LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred=logreg.predict(x_test)
    print("Accuracy",100*sklearn.metrics.accuracy_score(y_test,y_pred))
    bmi=float(BMI)
    print(age,cigsPerDay,totChol,sysBP,bmi,pulseRate,glucose)
    x=np.array([age,cigsPerDay,totChol,sysBP,bmi,pulseRate,glucose]).reshape(1,-1)
    print(x)
    prediction=logreg.predict(x)
    msg['To']=m
    server=smtplib.SMTP_SSL('smtp.gmail.com',465)
    server.login("shreeshyleshronaldo@gmail.com","slslyridpjrffziq")
    if prediction==0:
        msg.set_content("ʏᴏᴜ ʜᴀᴠᴇ ɴᴏ ʀɪꜱᴋ ᴏꜰ ɢᴇᴛᴛɪɴɢ ʜᴇᴀʀᴛ ᴅɪꜱᴇᴀꜱᴇ.ꜰᴏʀ ꜱᴀꜰᴇʀ ꜱɪᴅᴇ,ᴄᴏɴꜱᴜʟᴛ ᴀ ᴅᴏᴄᴛᴏʀ")
        server.send_message(msg)
        return render_template("negative.html")
        
    else:
        msg.set_content("ʏᴏᴜ ʜᴀᴠᴇ ʀɪꜱᴋ ᴏꜰ ɢᴇᴛᴛɪɴɢ ʜᴇᴀʀᴛ ᴅɪꜱᴇᴀꜱᴇ. ᴋɪɴᴅʟʏ ᴄᴏɴꜱᴜʟᴛ ᴀ ᴅᴏᴄᴛᴏʀ.")
        server.send_message(msg)
        return render_template('positive.html')
    
@app.route('/fit',methods=['POST'])
def fit():
    global sex
    global age
    global currentSmoker
    global cigsPerDay
    global BPMeds
    global prevalentHyp
    global prevalentStroke
    global diabetes
    global totChol
    global sysBP
    global diaBP
    global BMI
    global glucose
    global m
    
    sex=int(request.form['sex'])
    age=int(request.form['age'])
    currentSmoker = int(request.form['currentSmoker'])
    cigsPerDay = int(request.form['cigsPerDay'])
    BPMeds = int(request.form['BPMeds'])
    prevalentStroke = int(request.form['prevalentStroke'])
    prevalentHyp = int(request.form['prevalentHyp'])
    diabetes = int(request.form['diabetes'])
    totChol = int(request.form['totChol'])
    sysBP = int(request.form['sysBP'])
    diaBP = int(request.form['diaBP'])
    BMI = float(request.form['BMI'])
    glucose = int(request.form['glucose'])
    m=request.form['mail'] 
    return render_template("temp1.html")
if __name__=="__main__":
    app.run(debug=False,port=7384)