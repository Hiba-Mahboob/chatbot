from chat import Query
from flask import Flask, jsonify, request
import json
import runpy

app=Flask(__name__)

qf=Query()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/chatbotquery/<string:q>')
def chatbot(q):
    res=qf.get_response(q)
    if res=="I do not understand..., but i am saving it to response in future":
        result={
            "status": False,
            "res":res
        }
    else:
        result={
            "status": True,
            "res":res
        }
    return jsonify(result)

@app.route('/addquery', methods=['GET','POST'])
def addquery():
    if request.method=='POST':
        content = request.get_json()
        dictionary={
            "tag": content["tag"],
            "patterns" : [
                content["pattern"]
            ],
            "responses" : [
                content["response"]
            ]
        }
        with open("./intents.json", "r+") as file:
            jsFile = json.load(file)
            temp=jsFile['intents']
            temp.append(dictionary)
            file.seek(0)
            json.dump(jsFile, file, indent=4)
        runpy.run_path(path_name='train.py')
        return jsonify("Query added successfully")

    

if __name__ == "__main__":
    app.run(debug=True, port=5000)

# conda activate base_clone