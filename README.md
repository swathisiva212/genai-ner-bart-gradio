## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
The challenge is to build an NER system capable of identifying named entities (e.g., people, organizations, locations) in text, using a pre-trained BART model fine-tuned for this task. The system should be interactive, allowing users to input text and see the recognized entities in real-time
### DESIGN STEPS:

### STEP 1: Fine-tune the BART model
Start by fine-tuning the BART model for NER tasks. This involves training the model on a labeled NER dataset with text data that contains named entities (e.g., people, places, organizations).

### STEP 2: Create an API for NER model inference
Develop an API endpoint that takes input text and returns the recognized entities using the fine-tuned BART model.

### STEP 3: Integrate the API with Gradio
Build a Gradio interface that takes user input, passes it to the NER model via the API, and displays the results as highlighted text with identified entities.

### PROGRAM:
```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import gradio as gr

# Load pre-trained BERT NER model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a pipeline for NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to process user input
def ner_function(text):
    entities = ner_pipeline(text)
    return "\n".join([f"{ent['word']} ({ent['entity']})" for ent in entities])

# Gradio Interface
iface = gr.Interface(
    fn=ner_function,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="Named Entities"),
    title="NER Demo with Pre-trained Model"
)

iface.launch()
```
### OUTPUT:

![image](https://github.com/user-attachments/assets/2936faa4-73ac-4e22-b91f-0886cf39c981)


### RESULT:
Thus, developed an NER prototype application with user interaction and evaluation features, using a fine-tuned BART model deployed through the Gradio framework.
