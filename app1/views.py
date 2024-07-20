from django.shortcuts import render
import joblib
import pandas as pd
from django.conf import settings

def index(request):
    if request.method == 'POST':
        field = request.POST['field']
        rank = request.POST['rank']

        if field != "select":
            try:
                model, label_encoder = joblib.load(settings.JOBLIB_MODEL_FILE_PATH)

                new_data = pd.DataFrame([[rank, field]], columns=['OPEN_HS', 'Program'])
                new_data['OPEN_HS'] = new_data['OPEN_HS'].astype(int)
                new_data['Program'] = label_encoder.transform(new_data['Program'])

                prediction = model.predict(new_data)

                param = {"pre":prediction, 'status' : 1, 'rank' : rank, "field" : field}
            except:
                param = {"pre":["Sorry, Model is unable to predict for given data."],'status' : 0 , 'rank' : rank, "field" : field}
        else:
            param = {"pre":["Select Proper option"], 'status' : 1, 'rank' : rank, "field" : field}
        return render(request,'index.html', param)
    return render(request,'index.html')

def home(request):
    return render(request,'index.html')
def about(request):
    return render(request,'about.html')

