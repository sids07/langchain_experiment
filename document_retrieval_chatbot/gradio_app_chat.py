import gradio as gr
from milvus_langchain_chat import respond

if __name__=="__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot()
                msg = gr.Textbox(label="Input Phrase")
                clear = gr.Button("Clear")
        btn = gr.Button("Generate")
        btn.click(respond, inputs=[msg,chatbot], outputs=[msg,chatbot])

    demo.launch()