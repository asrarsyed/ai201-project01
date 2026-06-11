import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from generate import ask


def handle_query(question: str):
    if not question.strip():
        return "", ""
    result = ask(question)
    return result["answer"], result["sources"]


with gr.Blocks(title="OMS Course Review Assistant") as demo:
    gr.Markdown("## OMS Course Review Assistant")
    gr.Markdown("Ask questions about Georgia Tech OMS courses based on real student reviews.")

    inp = gr.Textbox(label="Ask about any OMS course", lines=2, placeholder="e.g. How hard is CS-7641 Machine Learning?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8, interactive=False)
    sources = gr.Textbox(label="Retrieved from", lines=5, interactive=False)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch(server_port=7860)
