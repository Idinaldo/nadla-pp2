# Prática 1 e 2 são necessárias para que isso funcione

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

link = 'https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/d4332a3056f44e1a1dec9600a31f21c8_boston.csv'
boston = pd.read_csv(link)

# criar dataframe Pandas
data = pd.DataFrame(data=boston)

#Ver o dataframe criado
data.head()

#descrever dataframe
data.describe()

#instalando o pandas profiling
!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

#import o profilereport
from pandas_profiling import ProfileReport

# executando o profile
profile = ProfileReport(data, title='Relatório - PAndas Profiling', html={'style':{'full_width':True}})

profile

y = data['MEDV']

x = data.drop(['TOWN', 'TRACT', 'LON', 'LAT', 'MEDV', 'CRIM', 'ZN', 'INDUS'], axis=1)

x.head()

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

print(f'X_train: número de linhas e colunas {X_train.shape}')
print(f'X_test: número de linhas e colunas {X_test.shape}')
print(f'y_train: número de linhas e colunas {y_train.shape}')
print(f'y_test: número de linhas e colunas {y_test.shape} ')

X_test.head()

predicoes = []

def retorna_baseline(n_quartos):
  return n_quartos * 10

for n_quartos in X_test['RM']:
  predicoes.append(retorna_baseline(n_quartos))

predicoes[:10]

df_results = pd.DataFrame()

df_results['valor_real'] = y_test.values

df_results['valor_predito_baseline'] = predicoes

df_results.head(10)

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_real, mode='lines+markers', name="Valor Real"))

fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_baseline, mode='lines+markers', name="Valor Predito Baseline"))

fig.show()


