from flask import Flask, request
from main import process_query, AIModelInstance

app = Flask(__name__)

@app.route('/', methods=['POST'])
def ask_ai():
    data = request.get_json()
    if 'question' not in data:
        return 'No questions asked'
    
    question = data['question']
    temperature = 0.0
    if 'temperature' in data:
        temperature = float(data['temperature'])

    continue_chat = False
    if 'continue_chat' in data:
        continue_chat = int(data['continue_chat'])

    llm_response = process_query(AIModelInstance.get_instance(), question, temperature)
    if not continue_chat:
        AIModelInstance.reset()

    return llm_response

if __name__ == '__main__':
    app.run()
