"""
Definition of views.
"""

from django.shortcuts import render
from django.http import HttpRequest
from django.template import RequestContext
from django.http import JsonResponse
from datetime import datetime, timedelta
#prediction
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier



def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Crime Analysis',
            'year':datetime.now().year,
        }
    )

def prediction(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/prediction.html',
        {
            'title':'Crime Prediction',
            'year':datetime.now().year,
        }
    )

#def about(request):
#    """Renders the about page."""
#    assert isinstance(request, HttpRequest)
#    return render(
#        request,
#        'app/about.html',
#        {
#            'title':'About',
#            'message':'Your application description page.',
#            'year':datetime.now().year,
#        }
#    )

def output(request):
    return JsonResponse([1, 2, 3, 4], safe=False)

def predict(request):

    #load scaler
    scaler2 = joblib.load('scaler.pkl') 
    #load model
    mlp2 = joblib.load('mlp.pkl') 
    
    # Get location features
    lat = request.GET['Lat'] #41.886340707	
    long = request.GET['Long']# -87.657908498
    locationFeatures = [lat, long]
    
    # Get date 
    dateFrom = datetime.strptime(request.GET['DateFrom'], '%Y-%m-%d')
    dateTo = datetime.strptime(request.GET['DateTo'], '%Y-%m-%d')
    
    # Declare list of result array
    results = []
    d = dateFrom
    delta = timedelta(days=1)
    while d <= dateTo:
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

        a = {  'date': d.strftime('%Y-%m-%d')
             , 'Person_Related': crime1
             , 'Property_Related': crime2
             , 'Drug_Related': crime3
             , 'Weapon_Related': crime4
             , 'Sexual_Exploitation': crime5
             , 'Other_Offense': crime6
             , 'Average': crimeAvg
             }

        # Add result to the list
        results.append(a)
        
        d += delta
    
        
    return JsonResponse(results, safe=False)
   
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