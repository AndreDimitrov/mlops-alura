from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from pyparsing import conditionAsParseAction
from textblob import TextBlob

# Inclusão - Modelo
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Carregar o modelo e dar permissaõ de leitura e gravar em uma variável.
colunas = ['tamanho','ano','garagem']
# ../ >> sobe uma pasta
modelo = pickle.load(open('../../models/modelo.sav', 'rb'))
# Fim inclusão Modelo

app = Flask('__name__')
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

# Endpoint
@app.route('/')
def home():
    return "Minha primeira API."

# Endpoint
@app.route('/sentimento/<frase>')
# Requisição de autorização para acesso
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

# Endpoint
@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0')


# Colocar todo o treinamento do modelo fora do endpoint

# Somente o predict que vai ser criado um endpoint e colocar dentro