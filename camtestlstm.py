import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x.unsqueeze(0), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def getFrames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_list = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames_list.append(frame)
        frame_count += 1

    video_capture.release()
    frames_array = np.array(frames_list)
    frames_array = np.transpose(frames_array, (0, 3, 1, 2))
    frames_tensor = torch.from_numpy(frames_array)
    return frames_tensor

def FeatureExtractor(frames_tensor):
    activation = {}
    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model = model.to(device)
    model.eval()
    preprocess = weights.transforms()
    vid = frames_tensor
    batch = preprocess(vid).unsqueeze(0).to(device)
    h2 = model.avgpool.register_forward_hook(getActivation('avgpool'))
    out = model(batch)
    feature_vector_tensor = activation['avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)
    feature_vector_np = feature_vector_tensor.detach().cpu().numpy()
    h2.remove()
    return feature_vector_np

def predictor(frames_tensor):
    feature_vector = FeatureExtractor(frames_tensor)
    tensor_features = torch.from_numpy(feature_vector)

    label_mapping = {
        0: "NoActivities",
        1: "antiClockwise",
        2: "blahblah",
        3: "clockwise",
        4: "come",
        5: "down",
        6: "like",
        7: "stop",
        8: "super",
        9: "up",
        10: "victory",
    }

    input_size = len(feature_vector[0])
    hidden_size = 128  # Changed from 64 to 128
    num_layers = 2  # You may need to adjust this if you still get an error
    num_classes = len(label_mapping) 
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load('LSTMmodel.pth'))
    model.eval()

    with torch.no_grad():
        output = model(tensor_features)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = label_mapping[int(predicted.numpy())]
    return predicted_label, confidence.item()


def save_frames(frames, output_dir):
    for i, frame in enumerate(frames):
        output_path = f"{output_dir}/frame_{i}.png"
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {i} to {output_path}")

vid = cv2.VideoCapture(0)
batch_iter = 0
frame_count = 0
frames_list = []
testDir = 'testFrames'

while True:
    ret, frame = vid.read()
    if not ret:
        break
    frames_list.append(frame)
    frame_count += 1

    if frame_count == 40:
        frames_array = np.array(frames_list)
        frames_array = np.transpose(frames_array, (0, 3, 1, 2))
        frames_tensor = torch.from_numpy(frames_array)
        prediction,confidence = predictor(frames_tensor)
        text = str(prediction)
        conf = str(confidence)
        print(text)
        print(conf)
        print('\n')
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(frame, text, (50, 50), font, font_scale, color, thickness)
        cv2.putText(frame, conf, (200, 50), font, font_scale, color, thickness)
        cv2.imshow('frame', frame)
        batch_iter+=1
        output_dir = testDir+'/'+text+str('_')+str(batch_iter)
        os.makedirs(output_dir,exist_ok=True)
        save_frames(frames_list, output_dir)
        frames_list = []
        frame_count = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

vid.release()
cv2.destroyAllWindows()
