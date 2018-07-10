#prediction
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import csv

def predict(period):
    lat = []
    long = []
    #dateFrom = datetime.strptime('2018-01-01', '%Y-%m-%d')
    #dateTo = datetime.strptime('2018-12-31', '%Y-%m-%d')
    dates = pd.date_range('20180101', periods=period,freq='D')
    
    # Get location features	
    airbnb = pd.read_csv('airbnb.csv', names = ["name", "neighbourhood", "latitude", "longitude"
                                                  , "room_type", "minimum_nights", "price"
                                                  , "price_per_night", "number_of_reviews", "last_review"
                                                  , "reviews_per_month"], header=1,sep=',')
    loca = airbnb[['latitude','longitude']]
    locationFeaturesList = loca.values.tolist()    
   
    #load scaler
    scaler2 = joblib.load('scaler.pkl') 
    #load model
    mlp2 = joblib.load('mlp.pkl') 
    # Declare list of result array
    results = []
    
    for locationFeatures in locationFeaturesList:            
        for d in dates:
            # Extract date features
            dateFeatures = ExtractDateFeature(d)
            # Prepare features
            features = [locationFeatures + dateFeatures]
            # Normalise features
            inputData = scaler2.transform(features)
            
            # Predicted crime probabilities at the date
            crimeProb = mlp2.predict_proba(inputData)[0].tolist()
            crime1 = float(format(crimeProb[0]*100, '.2f'))
            crime2 = float(format(crimeProb[1]*100, '.2f'))
            crime3 = float(format(crimeProb[2]*100, '.2f'))
            crime4 = float(format(crimeProb[3]*100, '.2f'))
            crime5 = float(format(crimeProb[4]*100, '.2f'))
            crime6 = float(format(crimeProb[5]*100, '.2f'))
            crimeAvg = float(format((crime1+crime2+crime3+crime4+crime5+crime6)/6, '.2f'))

            a = {  'Person_Related': crime1
                 , 'Property_Related': crime2
                 , 'Drug_Related': crime3
                 , 'Weapon_Related': crime4
                 , 'Sexual_Exploitation': crime5
                 , 'Other_Offense': crime6
                 , 'Average': crimeAvg
                 }
            # Add result to the list
            results.append(a)
    
    df = pd.DataFrame(results)    
    
    #save scaler
    joblib.dump(df, 'avgDataframe05012018.pkl')    
    #load scaler
    #df2 = joblib.load('avgDataframe.pkl') 

    #with open('AvgResults.csv', 'wb') as f:  # Just use 'w' mode in 3.x
    #    w = csv.DictWriter(f, results.keys())
    #    w.writeheader()
    #    w.writerow(results)
    print("End..")    
    return df

#### private function
def ExtractDateFeature(inputDate):
    day = inputDate.day
    mon = inputDate.month
    dow = inputDate.isoweekday()    

    if mon >=3 and mon <=5:
        season = 1
    elif mon >=6 and mon <=8:
        season = 2
    elif mon >=9 and mon <=11:
        season = 3
    else:
        season = 4

    # Setup features
    dateFeatures = [day,mon,dow,season]

    return dateFeatures

df = predict(365)
df["Average"].mean()
df["Person_Related"].mean()
df["Property_Related"].mean()
df["Drug_Related"].mean()
df["Weapon_Related"].mean()
df["Sexual_Exploitation"].mean()
df["Other_Offense"].mean()