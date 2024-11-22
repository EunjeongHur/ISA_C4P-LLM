# Importing the necessary classes from the http.server module
# Importing the JSON module for handling JSON data
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# Importing classes from the transformers library to load the model and tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set the model directory where the downloaded model is stored
model_directory = "../bart-large-cnn"
# Load the tokenizer from the specified model directory
tokenizer = AutoTokenizer.from_pretrained(model_directory)
# Load the seq2seq model from the specified model directory
model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)


# Define a request handler class that will handle incoming HTTP requests
class RequestHandler(BaseHTTPRequestHandler):
    # Define the method to handle POST requests
    def do_POST(self):
        # Get the length of the content in the request
        content_length = int(self.headers["Content-Length"])
        # Read the data sent in the POST request
        post_data = self.rfile.read(content_length)
        # Parse the JSON data into a Python dictionary
        data = json.loads(post_data)
        # Extract the input text from the parsed data; default to an empty string if not found
        input_text = data.get("input", "")

        print(f"Received input: {input_text}")

        # Tokenize the input text and prepare it as input for the model
        inputs = tokenizer(input_text, return_tensors="pt")
        # Generate a summary using the model with specific parameters
        outputs = model.generate(
            **inputs,
            max_length=3000,  # Maximum length of the generated summary
            min_length=10,  # Minimum length of the generated summary
            length_penalty=2.0,  # Penalty for longer sequences to encourage brevity
            num_beams=4,  # Number of beams for beam search
            early_stopping=True,  # Stop the generation early if certain conditions are met
        )
        # Decode the generated output tensor to get the summary in text format
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Send a 200 OK response back to the client
        self.send_response(200)
        # Set the content type of the response to JSON
        self.send_header("Content-Type", "application/json")
        # End the headers section of the response
        self.end_headers()
        # Send the summary back to the client as a JSON object
        self.wfile.write(json.dumps({"summary": summary}).encode("utf-8"))


# Function to run the HTTP server
def run(server_class=HTTPServer, handler_class=RequestHandler, port=5050):
    # Set the server address to listen on all interfaces at the specified port
    server_address = ("", port)
    # Create an instance of the HTTP server with the specified address and request handler
    httpd = server_class(server_address, handler_class)
    # Print a message indicating the server is starting
    print(f"Serving on port {port}...")
    # Start the server and keep it running indefinitely to handle requests
    httpd.serve_forever()


# Check if the script is being run directly (not imported)
if __name__ == "__main__":
    # Call the run function to start the server
    run()
