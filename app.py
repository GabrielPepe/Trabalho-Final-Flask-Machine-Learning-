from flask import Flask, render_template, request
import machine_learning

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
  # se o usuario der submit, o if é ativado, e mostrará na tela a matriz de confusão com as métricas
  if request.method == "POST":
    # req está pegando todos os valores que estão no forms, com base no name do html deles e instanciando cada valor em uma variavel
    req = request.form
    print(req)
    parametro1 = (req["parametro1"])
    classificador = (req["classificador"])
    randoms = (req["randoms"])
    # chama a função do machine learning, instancia as variaveis necessarias na função e retorna as métricas para coloca-las na tela
    accuracy, precision, recall, f1 = machine_learning.machine_learning(p1=parametro1, p2='', classificador=classificador, randoms= randoms)
    return render_template("index.html", img="img.png", exibir_imagem=True, accuracy=accuracy, precision=precision, f1=f1, recall=recall)
  return render_template("index.html")

@app.route("/mlp", methods=['POST', 'GET'])
def mlp():
  if request.method == "POST":
    req = request.form
    print(req)
    parametro1 = (req["parametro1"])
    parametro2 = (req["parametro2"])
    classificador = (req["classificador"])
    randoms = (req["randoms"])
    accuracy, precision, recall, f1 = machine_learning.machine_learning(p1=parametro1, p2=parametro2, classificador=classificador, randoms= randoms)
    return render_template("mlp.html", img="img.png", exibir_imagem=True, accuracy=accuracy, precision=precision, f1=f1, recall=recall)
  return render_template("mlp.html")

@app.route("/randomForest", methods=['POST', 'GET'])
def randomForest():
  if request.method == "POST":
    req = request.form
    parametro1 = (req["parametro1"])
    parametro2 = (req["parametro2"])
    classificador = (req["classificador"])
    randoms = (req["randoms"])
    accuracy, precision, recall, f1 = machine_learning.machine_learning(p1=parametro1, p2=parametro2, classificador=classificador, randoms= randoms)
    return render_template("rf.html", img="img.png", exibir_imagem=True, accuracy=accuracy, precision=precision, f1=f1, recall=recall)
  return render_template("rf.html")

@app.route("/decisionTree", methods=['POST', 'GET'])
def decisionTree():
  if request.method == "POST":
    req = request.form
    parametro1 = (req["parametro1"])
    parametro2 = (req["parametro2"])
    classificador = (req["classificador"])
    randoms = (req["randoms"])
    accuracy, precision, recall, f1 = machine_learning.machine_learning(p1=parametro1, p2=parametro2, classificador=classificador, randoms= randoms)
    return render_template("dt.html", img="img.png", exibir_imagem=True, accuracy=accuracy, precision=precision, f1=f1, recall=recall)
  return render_template("dt.html")

if __name__ == "__main__":
  app.run(debug=True)
