import gradio as gr
from transformers import pipeline 
from PIL import Image

def remove_background(image):
  pipe = pipeline('image-segmentation', model='briaai/RMBG-1.4',trust_remote_code=True)
  pillow_mask = pipe(image, return_mask=True)
  pillow_image = pipe(image)
  return pillow_image

app = gr.Interface(
    fn= remove_background, #função
    inputs= gr.components.Image(type='pil'), #entrada (anexo do usuário)
    outputs = gr.components.Image(type='pil'),#saida (download da imagem sem fundo no formato PNG)
    title= 'Remoção de background de Imagens',
    description= 'Envie uma imagem e veja o background removido automaticamente. A imagem resultante será no formato PNG.'
)

if __name__ == "__main__":
    app.launch(share=True) #Share=True gera um link que pode ser compartilhado