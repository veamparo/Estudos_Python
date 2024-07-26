#Importação das bibliotecas que serão usadas
import sys
import pandas as pd
import shap
from skopt.space import Real, Categorical, Integer
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve, select_threshold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#leitura da tabela com os dados dos 273 pacientes

data = pd.read_excel(r'C:\Users\mhdfr\Downloads\DATA_19.03.22.xlsx')

#Separação da coluna com os rótulos da presença de doença
target = data['Doença existente']
data = data.drop(columns = ['Doença existente'])

#transformação para as variáveis categóricas

encoder = OneHotEncoder(categories = 'auto', sparse = False)

#Aqui transformamos a variável categórica Sexo em duas colunas, uma chamada Male e outra Female.
x = data['Sexo'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["Male"] = x[:,0]
data["Female"] = x[:,1]

#Repetimos o processo analogamente para as outras variáveis.

x = data['Resting electrocardiographic results'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["Resting electrocardiographic results tipo_0"] = x[:,0]
data["Resting electrocardiographic results tipo_1"] = x[:,1]
data["Resting electrocardiographic results tipo_2"] = x[:,2]


x = data['Chest pain type'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["Chest pain_type_1"] = x[:,0]
data["Chest pain_type_2"] = x[:,1]
data["Chest pain_type_3"] = x[:,2]
data['Chest pain_type_4'] = x[:,3]

x = data['Number of major vessels (0-3) colored by flourosopy'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["Major vessels_type_0"] = x[:,0]
data["Major vessels_type_1"] = x[:,1]
data["Major vessels_type_2"] = x[:,2]
data['Major vessels_type_3'] = x[:,3]

x = data['The slope of the peak exercise ST segment'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["slope_type_1"] = x[:,0]
data["slope_type_2"] = x[:,1]
data["slope_type_3"] = x[:,2]

data = data.drop(columns = ['Resting electrocardiographic results'])
data = data.drop(columns = ['Sexo'])
data = data.drop(columns = ['Chest pain type'])
data = data.drop(columns = ['The slope of the peak exercise ST segment'])
data = data.drop(columns = ['Number of major vessels (0-3) colored by flourosopy'])

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.30, random_state = 42,stratify=data['Major vessels_type_0'])

#Escalonamento das variáveis númericas, tanto para os dados de teste como de treino
scaler = MinMaxScaler()

features = ['resting blood pressure', 'Serum cholestoral in mg/dl', 'Idade', 'Maximum heart rate achieved','Oldpeak = ST depression induced by exercise relative to rest']

bases = [x_train, x_test]

def Scaler(features):
    for base in bases:
        for feature in features:
            t = base[feature].values
            t = t.reshape(len(t), 1)
            t = scaler.fit_transform(t)
            base[feature] = t[:, 0]

Scaler(features)

clf = CatBoostClassifier(bagging_temperature= 0.8123959883573634,border_count=45,depth=5,iterations=805,l2_leaf_reg=17,learning_rate=0.01552065618292981, random_strength= 0.0361414296804267, silent=True
)

#utilização de validação cruzada no treino
kfold = KFold(n_splits = 10,random_state=42,shuffle=True)

#aplicação do algortimo utilizando validação cruzada
results = cross_val_score(clf, x_train, y_train, cv = kfold)

# impressão dos resultados da média de acurácia e desvio padrão.
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

catboost_pool = Pool(x_train, y_train)
clf.fit(x_train, y_train)
roc_curve_values = get_roc_curve(clf, catboost_pool)

boundary = select_threshold(clf,
                            curve=roc_curve_values,
                            FNR=0.001)
print(boundary)


clf.fit(x_train, y_train)

#predizemos nos dados de teste os pacientes que segundo o modelo podem ter doença cardiaca, note que o limiar de decisão
# é de 0.43, ou seja, pacientes que pontuem acima desse valor, marcamos como portadores de doença.

y_pred = (clf.predict_proba(x_test)[:,1] >=0.43)

from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

print("Acurracy Score :",accuracy_score(y_test, y_pred))

print("Auc Score : ",roc_auc_score(y_test, y_pred))

print("Recall Score : ",recall_score(y_test, y_pred,
                                           pos_label='positive',average='micro'))

probs = clf.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()