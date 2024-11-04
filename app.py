# from http.server import BaseHTTPRequestHandler, HTTPServer
# import json
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# model_name = "facebook/bart-large-cnn"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# class RequestHandler(BaseHTTPRequestHandler):
#     def do_POST(self):
#         content_length = int(self.headers['Content-Length'])
#         post_data = self.rfile.read(content_length)
#         data = json.loads(post_data)
#         input_text = data.get("input", "")

#         inputs = tokenizer(input_text, return_tensors="pt")
#         outputs = model.generate(**inputs, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
#         summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         self.send_response(200)
#         self.send_header('Content-Type', 'application/json')
#         self.end_headers()
#         self.wfile.write(json.dumps({"summary": summary}).encode('utf-8'))

# def run(server_class=HTTPServer, handler_class=RequestHandler, port=5000):
#     server_address = ('', port)
#     httpd = server_class(server_address, handler_class)
#     print(f'Serving on port {port}...')
#     httpd.serve_forever()

# if __name__ == "__main__":
#     run()


# If you download the model in the .. directory, you can run the server with the following command:
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_directory = "../bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        input_text = data.get("input", "")

        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"summary": summary}).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Serving on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
