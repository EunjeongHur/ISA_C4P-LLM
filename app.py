# from flask import Flask, request, jsonify
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = Flask(__name__)

# model_name = "facebook/bart-large-cnn"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# @app.route("/", methods=["POST"])
# def generate_text():
#     data = request.json
#     prompt = data.get("prompt", "")
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=50)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({"response": generated_text})

# if __name__ == "__main__":
#     app.run(port=5000)


from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route("/", methods=["POST"])
def generate_text():
    data = request.json
    input_text = data.get("input", "")
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": generated_text})

if __name__ == "__main__":
    app.run(port=5000)
