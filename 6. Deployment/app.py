#from joblib import load
from flask import Flask, request, render_template
#from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
app = Flask(__name__)

#import os
#print("Current Working Directory:", os.getcwd())

# Load the model and tokenizer
model_path = r'5. Modelling\bert_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Class labels
labels = {
    0: "The text is normal and shows no indicators of concern.",
    1: "The text indicates warning signs of a mass shooter that might warrant closer observation.",
    2: "The text is a direct threat, requiring immediate attention and monitoring."
}   
    
#define the route
#app.route = .xx.com
@app.route('/')
def home():
    # Initially, pass an empty string for prediction
    return render_template('index.html', prediction="") # <-- Change here

#def to_predict(text):
@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        
        '''
        # Load the vectorizer
        vectorizer = load(r'5. Modelling\tfidf_vectorizer.pkl')
        # Transform the input text
        text_vectorized = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0] # Extract the scalar value
        '''
        # Tokenize the input text
        
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        result = labels.get(predicted_class, "Unknown prediction.")
            
        # Render the result in the template
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")
    
if __name__ == '__main__':
    app.run()
    