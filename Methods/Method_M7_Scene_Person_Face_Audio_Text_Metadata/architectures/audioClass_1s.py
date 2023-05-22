import torch
import torch.nn as nn
from .vggish2 import VGGish

class audioConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        audio_model = VGGish(),
        audio_output=128,
        lstm_size = 128,
        num_layers_lstm = 1,
        dropout_p = 0.2
        ):

        super(audioConcat, self).__init__()

        self.audio_module = audio_model.to(torch.device('cuda:0'))

        #self.audio_module.load_state_dict(torch.load('pytorch_vggish.pth')) #load pretrained weights
        
        self.audio_module.eval()
        
        self.audio_module = self.audio_module.to(torch.device('cuda:0'))

        self.audio_layer = torch.nn.Linear(
                in_features=audio_output,
                out_features=audio_output
        )

        self.fc = torch.nn.Linear(
            in_features=audio_output, 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, audioOneSecond): #please give us input already embedded vggish

        audio_features = torch.empty((len(audioOneSecond), 15, 128)).to(torch.device('cuda:0'))
        audio = audioOneSecond.reshape(-1, 1, 96, 64)
        #print('Shape after reshape: ', audio.shape)
        vggEmb = self.audio_module(audio)
        vggEmb = vggEmb.reshape(-1, 15, vggEmb.shape[-1])

        audio_features = vggEmb.mean(dim=1)

        output_audio = self.dropout(torch.nn.functional.relu(
            self.audio_layer(audio_features)
        ))
        logits = self.fc(output_audio)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred
    
