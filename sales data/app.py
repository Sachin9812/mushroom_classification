import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import OneHotEncoder

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
model = pickle.load(open('Pickle_RF_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #int_features = [float(x) for x in request.form.values()]
    int_features = [str(x) for x in request.form.values()]

    import pandas as pd
    test = pd.read_csv('mushroom_final_train.csv', index_col='Unnamed: 0')
    onehotencoder = OneHotEncoder(handle_unknown='ignore')

    X = onehotencoder.fit_transform(test).toarray()
    xx = onehotencoder.transform([int_features]).toarray()
    print(xx)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    x_test = pca.fit_transform(X)
    final_features = pca.transform(xx)
    print(final_features)

    #final_features = [np.array(x_test)]
    prediction = model.predict(final_features)
    print(int_features)
    output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='Mushrooms are {}'.format(output))
    if (output == 1):
        return render_template('index.html', prediction_text='Mushrooms are poisonous')
    elif (output == 0):
        return render_template('index.html', prediction_text='Mushrooms are edible')


if __name__ == "__main__":
    app.run(debug=True)