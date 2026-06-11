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
    gr.Markdown("# OMS Course Review Assistant", elem_id="title")
    gr.Markdown(
        "Ask anything about Georgia Tech OMS courses. Answers are grounded in real student reviews from [OMSHub](https://www.omshub.org/).",
        elem_id="subtitle",
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Ask about any OMS course",
            placeholder="e.g. How hard is CS-7641 Machine Learning?",
            lines=2,
            scale=9,
        )
        btn = gr.Button("Ask", variant="primary", scale=1, elem_id="ask-btn")

    answer = gr.Textbox(label="Answer", lines=10, interactive=False)
    sources = gr.Textbox(label="Retrieved from", lines=4, interactive=False)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        #title { text-align: center; padding: 1rem 0 0.25rem; }
        #subtitle { text-align: center; color: #6b7280; margin-bottom: 1rem; }
        """,
    )
