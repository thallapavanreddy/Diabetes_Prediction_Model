## Diabetes-Prediction-Model-Flask-Deployment
This is a project to deploye Prediction Model on production using Flask API

### Prerequisites
In Working environment must have Scikit Learn, Pandas and Flask installed.

'''
pip install requirments.txt
'''


To Check libraies once 
'''
pip list
'''

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict Diabetes-Prediction-Model absed on trainign data in 'diabetes.csv' file.
2. app.py - This contains Flask APIs that receives test person details through GUI, computes the precited value based on our model and returns it.
3. templates - This folder contains the HTML template to allow user to enter person detail and displays the predicte diabetes.

### Running the project
1. In the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file diabetesML.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](http://www.thepythonblog.com/wp-content/uploads/2019/02/Homepage.png)

Enter valid numerical values in all input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
