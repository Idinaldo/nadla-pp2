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

# Salvando o relatório no disco
profile.to_file(output_file="Relatorio.html")

"""

```
# Isto está formatado como código
```

# Aula 2
"""

# checando se há valores nulos
data.insull().sum()
  TOWN 0
  TRACT 0
  LON 0
  LAT 0
  MEDV 0
  CRIM 0
  ZN 0
  INDUS 0
  CHAS 0
  NOX 0
  RM 0
  AGE 0
  DIS 0
  RAD 0
  TAX 0
  PTRATIO 0
  dtype:
  int64

# Descrição estatística
data.describe()

#calculando correlação (deu erro)
correlacoes = data.corr()

#calculando correlação
# Select only numerical features for correlation calculation
numerical_features = data.select_dtypes(include=np.number).columns
correlacoes = data[numerical_features].corr()

# Criandográfico de calor das correlações
plt.figure(figsize=(16, 6))
sns.heatmap(data=correlacoes, annot=True)

#importando plot.ly
import plotly.express as px

# RM vs MEDV (número de quartos e valor médio do imóvel)
fig = px.scatter(data, x=data.RM, y=data.MEDV)
fig.show()

#PTRATIO vs MEDV (percentual de proporção de alunos para professores)
fig = px.scatter(data, x=data.PTRATIO, y=data.MEDV)
fig.show()

#estatísticas descritiva da variável RM
data.RM.describe()

# visualizando a distribuição da variável RM
import plotly.figure_factory as ff
labels = ['Distribuição de variável RM (número de quartos)']
fig = ff.create_distplot([data.RM], labels, bin_size=.2)
fig.show()

#visualizando outliers na variável RM
import plotly.exceptions as px

fig = px.box(data, y='RM')
fig.update_layout(width=800, height=800)
fig.show()

#visualizando outliers na variável RM
import plotly.express as px # Change this line to import plotly.express

fig = px.box(data, y='RM')
fig.update_layout(width=800, height=800)
fig.show()

#estatística descritiva da variável MEDV
data.MEDV.describe()

#visualizando a distribuiçao da variável MEDV
import plotly.figure_factory as ff
labels = ['Distribuição de variável MEDV (valor médio do imóvel)']
fig = ff.create_distplot([data.MEDV], labels, bin_size=.2)
fig.show()

#carrega o método stats da scipy
from scipy import stats

# imprime o coeficiente de pearson
stats.sknew(data.MEDV)

#carrega o método stats da scipy
from scipy import stats
# %%
# imprime o coeficiente de pearson
# Use stats.skew instead of stats.sknew to calculate skewness
stats.skew(data.MEDV)

# histogram da variável MEDV (variável alvo)
fig = px.histogram(data, x='MEDV', nbins=50, opacity=0.50)
fig.show()

#visualisando outliers na variável MEDV
import protly.express as px

fig = px.box(data, y='MEDV')
fig.update_layout(width=800, height=800)
fig.show()

#visualisando outliers na variável MEDV
import plotly.express as px # Corrected the import statement

fig = px.box(data, y='MEDV')
fig.update_layout(width=800, height=800)
fig.show()

data[['RM', 'MEDV', 'PTRATIO']].describe()

data[['RM', 'MEDV','PTRATIO']].nlargest(20, 'MEDV')
top20 = data.nlargest(20, 'MEDV').index
top20
data.drop(top20, inplace=True)

#imprimindo os 16 maiores valores de MEDV
data[['RM', 'PTRATIOMEDV', 'MEDV']].nlargest(16, 'MEDV')

#imprimindo os 16 maiores valores de MEDV
data[['RM', 'PTRATIO', 'MEDV']].nlargest(16, 'MEDV') # Changed 'PTRATIOMEDV' to 'PTRATIO'

# filtra os top 16 maiores registro da coluna MEDV
top16 = data.nlargest(16, 'MEDV').index

#remove os valores listados em top16
data.drop(top16, inplace=True)

#visualisando a distribuição da variável MEDV
import protly.figure_factory as ff
labels = ['Distribuição de variável MEDV (número de quartos)']
fig = ff.create_distplot([data.MEDV], labels, bin_size=.2)
fig.show()

#visualisando a distribuição da variável MEDV
import plotly.figure_factory as ff # Corrected the import statement to 'plotly'
labels = ['Distribuição de variável MEDV (número de quartos)']
fig = ff.create_distplot([data.MEDV], labels, bin_size=.2)
fig.show()

# histogram da variável MEDV (variável alvo)
fig = px.histogram(data, x='MEDV', nbins=50, opacity=0.50)
fig.show()

#converte os dados
data.RM = data.RM.astype(int)

#imprime o coeficiente de pearson
stats.skew(data.MEDV)

#estatística descritiva da coluna cumero de quartos
data.RM.describe()

#definindo a regra para categorizar os dados
categorias = []

#alimenta a lista categorias
for i in data.RM.items(): # Use .items() instead of .iteritems()
  valor = (i[1])
  if valor <= 4:
    categorias.append('pequeno')
  elif valor < 7:
    categorias.append('medio')
  else:
    categorias.append('grande')

# cria a coluna categorias
data['categorias'] = categorias

# imprime a contagem de categorias
data.categorias.value_counts()

# agrupa as categorias e calcula as médias
medias_categorias = data.groupby(by='categorias')['MEDV'].mean()

#visualizando a variável medias_categorias
medias_categorias

#criando o dicionario com chaves medio, grande e pequeno e seus valores
dic_baseline = {'medio': medias_categorias[1], 'grande': medias_categorias[0], 'pequeno': medias_categorias[2]}

#imprime o dicionario
dic_baseline

# cria a funçaõ retorna baseline
def retorna_baseline(num_quartos):
  if num_quartos <= 4:
    return dic_baseline['pequeno']
  elif num_quartos < 7:
    return dic_baseline['medio']
  else:
    return dic_baseline['grande']

#chama a função retorna baseline
retorna_baseline(3)

for i, n_quartos in data.RM.items(): # Use .items() to iterate through the Series
  print('Número de quartos é: {}, valor médio: {}'.format(n_quartos, retorna_baseline(n_quartos)))

#imprime as 5 primeiras linas do dataframe
data.head()
