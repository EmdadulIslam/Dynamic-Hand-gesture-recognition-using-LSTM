from torchvision.io.video import read_video
import torch
import numpy as np
import os
from glob import glob
from torchvision.models.video import r3d_18, R3D_18_Weights

def VideoEncode(video_path):
    activation = {}
    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    save_path = "\\".join(video_path.split('\\')[:-1])+'\\Res3d_feature.npy'

    if os.path.exists(save_path):
        print("The file Res3d_feature.npy exists in the specified location.")
        return "None"
    else:
        vid, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
        
        total_frames = vid.shape[0]
        indices = np.linspace(0, total_frames - 1, 40, dtype=int)
        vid = vid[indices]

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model = model.to(device)
        model.eval()

        preprocess = weights.transforms()

        if isinstance(vid, np.ndarray):
            vid = torch.from_numpy(vid)

        batch = preprocess(vid).unsqueeze(0).to(device)

        print("\n Batch shape we got: ", batch.shape)

        h2 = model.avgpool.register_forward_hook(getActivation('avgpool'))
        out = model(batch)

        feature_vector_tensor = activation['avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)
        feature_vector_np = feature_vector_tensor.detach().cpu().numpy()

        np.save(save_path, feature_vector_np)
        h2.remove()

        return feature_vector_np

if __name__ == '__main__':
    video_path = glob("F:\\HGD2_full\\*\\*\\*\\*.mp4")

    for video in video_path:
        print("...............")
        print("\n\nFor Video: ", video)
        encoded_featureVector = VideoEncode(video)
