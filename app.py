import gradio as gr
import lightning as L
from lightning.app.components.serve import ServeGradio


class LitGradio(ServeGradio):

    inputs = gr.inputs.Textbox(label="Enter text to generate scientific content")
    outputs = gr.outputs.Textbox(label="prediction")
    examples = [["what is Newton's law of motion?"]]

    def __init__(self):
        super().__init__(cloud_compute=L.CloudCompute("gpu"))

    def predict(self, text) -> str:
        return self.model.generate(text)

    def build_model(self):
        import galai as gal

        model = gal.load_model("mini", num_gpus=1)
        return model


app = L.LightningApp(LitGradio())
