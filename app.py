import torch
from torch import nn
import gradio as gr
from torchvision.transforms import Resize, ToTensor, Compose
from torch.nn.functional import softmax

class myCNN(nn.Module):
    def __init__(self, input_channels, classes) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3,3), padding='valid', bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding='valid', bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.MaxPool2d((2,2)),
                                    nn.Dropout2d(0.4))

        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding='valid', bias=False),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='valid', bias=False),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.MaxPool2d((2,2)),
                                    nn.Dropout2d(0.4))
        self.flat = nn.Flatten()

        self.fc1 = nn.Sequential(nn.Linear(3200, 512),
                                 nn.ReLU(),
                                 nn.Dropout1d(0.5))

        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU())

        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        flat = self.flat(layer6)
        fc1 = self.fc1(flat)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        return fc3
    

device = 'gpu' if torch.cuda.is_available() else 'cpu'

model_state = torch.load("myCNN_states.pt", map_location=device, weights_only=False)
input_shape = model_state['input_shape']
cls_to_idx = model_state['labels_encoder']
idx_to_cls = {value:key for key,value in cls_to_idx.items()}

pre_processor = Compose([Resize(input_shape[1:]),
                         ToTensor()])

model = torch.load("myCNN.bin",
                    map_location=device,
                    weights_only=False)

def post_processor(raw_output):
    softmax_output = softmax(raw_output, -1)
    values, indices = torch.max(softmax_output, -1)
    return idx_to_cls[indices.item()].capitalize(), round(values.item(), 2)


@torch.no_grad
def lunch(raw_input):
    input = pre_processor(raw_input)
    output = model(input.unsqueeze(0))
    return post_processor(output)

custom_css ='.gr-button {background-color: #bf4b04; color: white;}'

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label='Input Image')
            gr.Text("Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck", label="Supported Classes:")
        with gr.Column():
            class_name = gr.Textbox(label="This is (a\\an)")
            confidence = gr.Textbox(label='Confidence')
            start_btn = gr.Button(value='Submit', elem_classes=["gr-button"])
    start_btn.click(fn=lunch, inputs=input_image, outputs=[class_name, confidence])

demo.launch()