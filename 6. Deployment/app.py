#from joblib import load
from flask import Flask, request, render_template
#from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from data_preprocessing.textcleanpipeline import clean_text_pipeline, TextCleaner
import os
app = Flask(__name__)

#import os
#print("Current Working Directory:", os.getcwd())

# Get the directory of the current script (app.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Step OUT of "6. Deployment" and into "5. Modelling/Bert_model"
model_path = os.path.abspath(os.path.join(base_dir, "..", "5. Modelling", "Bert_model"))

tokenizer = BertTokenizer.from_pretrained(model_path,  local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path,  local_files_only=True)
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
        raw_text = request.form['text']
        
        cleaned_text = clean_text_pipeline(raw_text)
        if cleaned_text == "invalid input":
            return render_template('index.html', prediction="Input cannot be processed.") #input cannot be processed
        
        inputs = tokenizer(
            cleaned_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
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
    