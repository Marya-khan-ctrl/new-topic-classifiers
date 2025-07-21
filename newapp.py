import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr

# Load model and tokenizer
labels = ["World", "Sports", "Business", "Sci/Tech"]
tokenizer = BertTokenizer.from_pretrained("bert_news_model")
model = BertForSequenceClassification.from_pretrained("bert_news_model")
model.eval()

# Prediction function
def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return labels[prediction]

# Interface
with gr.Blocks(css="""
    body { background-color: #f9f9f9; }
    .title { font-size: 1.8em; font-weight: 600; color: #2c3e50; margin: 10px 0; }
    .subtitle { font-size: 1em; color: #555; margin-bottom: 15px; }
    .gr-button { font-size: 0.85em !important; padding: 0.4em 0.8em !important; border-radius: 6px; }
    .output-label { font-weight: bold; font-size: 1em; color: #1a2b4c; }
    #news-gif img { max-height: 160px !important; width: auto !important; margin-bottom: 10px; }
""") as demo:
    
    with gr.Column():
        gr.Markdown("<div class='title'>🧠 News Topic Classifier</div>")
        gr.Markdown("<div class='subtitle'>Classify headlines as <b>World</b>, <b>Sports</b>, <b>Business</b>, or <b>Sci/Tech</b>.</div>")

        gr.Image("https://media.giphy.com/media/l0HUpt2s9Pclgt9Vm/giphy.gif", elem_id="news-gif", show_label=False)

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(lines=2, max_lines=3, placeholder="e.g. NASA launches new space telescope...", label="Enter News Text")
                submit_btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=1):
                output_label = gr.Textbox(label="Predicted Topic", interactive=False, elem_classes="output-label")
        
        submit_btn.click(fn=classify_news, inputs=input_text, outputs=output_label)

demo.launch()
