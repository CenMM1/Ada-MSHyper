import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
import mediapipe as mp
import scipy.ndimage
import gdown

# --- Copied from libreface.Facial_Expression_Recognition.models.resnet18 ---
class ResNet(nn.Module):
    def __init__(self, opts):
        super(ResNet, self).__init__()
        self.fm_distillation = opts.fm_distillation
        self.dropout = opts.dropout
        resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        resnet18_layers = list(resnet18.children())[:-1]
        self.encoder = nn.Sequential(*resnet18_layers)
        self.classifier = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Dropout(p=self.dropout),
                nn.Linear(128, opts.num_labels),
                nn.Sigmoid()
    )
   
    def forward(self, images):
        batch_size = images.shape[0]
        features = self.encoder(images).reshape(batch_size, -1)
        labels = self.classifier(features)
        if not self.fm_distillation:
            return labels
        else:
            return labels, features

# --- Copied from libreface.utils ---
def download_weights(drive_id, model_path):
    model_dir = "/".join(model_path.split("/")[:-1])
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Downloading model weights - {model_path}...")
        gdown.download(id=drive_id, output=model_path)
        if not os.path.exists(model_path):
            print("Error occured in downloading...")
    return model_path

# --- Copied from libreface.detect_mediapipe_image ---
def image_align(img, face_landmarks, output_size=256,
        transform_size=512, enable_padding=True, x_scale=1,
        y_scale=1, em_scale=0.1, alpha=False, pad_mode='const'):

  lm = np.array(face_landmarks)
  lm[:,0] *= img.size[0]
  lm[:,1] *= img.size[1]

  lm_eye_right      = lm[0:16]  
  lm_eye_left     = lm[16:32]  
  lm_mouth_outer   = lm[32:]  
  # lm_mouth_inner   = lm[60 : 68]  # left-clockwise
  lm_mouth_outer_x = lm_mouth_outer[:,0].tolist()
  left_index = lm_mouth_outer_x.index(min(lm_mouth_outer_x))
  right_index = lm_mouth_outer_x.index(max(lm_mouth_outer_x))
  # print(left_index,right_index)
  # Calculate auxiliary vectors.
  eye_left     = np.mean(lm_eye_left, axis=0)
  # eye_left[[0,1]] = eye_left[[1,0]]
  eye_right    = np.mean(lm_eye_right, axis=0)
  # eye_right[[0,1]] = eye_right[[1,0]]
  eye_avg      = (eye_left + eye_right) * 0.5
  eye_to_eye   = eye_right - eye_left
  # print(lm_mouth_outer)s
  mouth_avg    = (lm_mouth_outer[left_index,:] + lm_mouth_outer[right_index,:])/2.0
  # mouth_avg[[0,1]] = mouth_avg[[1,0]]
  
  eye_to_mouth = mouth_avg - eye_avg
  # Choose oriented crop rectangle.
  x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
  x /= np.hypot(*x)
  x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
  x *= x_scale
  y = np.flipud(x) * [-y_scale, y_scale]
  c = eye_avg + eye_to_mouth * em_scale
  quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
  qsize = np.hypot(*x) * 2

  # Shrink.
  shrink = int(np.floor(qsize / output_size * 0.5))
  if shrink > 1:
    rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
    img = img.resize(rsize, Image.ANTIALIAS)
    quad /= shrink
    qsize /= shrink

  # Crop.
  border = max(int(np.rint(qsize * 0.1)), 3)
  crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
  crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
  if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
    img = img.crop(crop)
    quad -= crop[0:2]

  # Pad.
  pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
  pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
  if enable_padding and max(pad) > border - 4:
    pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
    if pad_mode == 'const':
      img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant', constant_values=0)
    else:
      img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
    h, w, _ = img.shape
    y, x, _ = np.ogrid[:h, :w, :1]
    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
    blur = qsize * 0.02
    img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
    img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
    img = np.uint8(np.clip(np.rint(img), 0, 255))
    if alpha:
      mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
      mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
      img = np.concatenate((img, mask), axis=2)
      img = Image.fromarray(img, 'RGBA')
    else:
      img = Image.fromarray(img, 'RGB')
    quad += pad[:2]

  img = img.transform((transform_size, transform_size), Image.Transform.QUAD,
            (quad + 0.5).flatten(), Image.Resampling.BILINEAR)

  out_image = img.resize((output_size, output_size), Image.Resampling.LANCZOS)
  # out_image = img

  return out_image

# --- Main Implementation ---

class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# FER ResNet18 Student Model with encode method
class FERResNet18Student(ResNet):
    def __init__(self, opts):
        super().__init__(opts)
        # Load weights
        ckpt_path = f'./weights_libreface/Facial_Expression_Recognition/weights/resnet.pt'
        download_weights(opts.weights_download_id, ckpt_path)
        checkpoints = torch.load(ckpt_path, map_location=opts.device, weights_only=True)['model']
        self.load_state_dict(checkpoints, strict=True)

    def encode(self, images):
        """
        Extract 512-d embeddings from global average pooling before fc.
        Input: [B, 3, 224, 224] aligned faces
        Output: [B, 512] embeddings
        """
        with torch.no_grad():
            self.eval()
            batch_size = images.shape[0]
            x = self.encoder(images)
            
            # Ensure we have [B, 512]
            if x.dim() == 4:
                # If output is [B, C, H, W], apply GAP
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                
            features = x.reshape(batch_size, -1)  # [B, 512]
            return features

# Function to load the model
def load_fer_model(device='cpu'):
    opts = ConfigObject({
        'seed': 0,
        'fm_distillation': True,
        'dropout': 0.1,
        'num_labels': 8,
        'weights_download_id': '1PeoPj8rga4vU2nuh_PciyX3HqaXp6LP7',
        'device': device
    })
    model = FERResNet18Student(opts).to(device)
    return model

# Preprocessing transform (same as libreface)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_face(image):
    """
    Preprocess aligned face image to tensor.
    Input: PIL Image (aligned 224x224)
    Output: torch.Tensor [3, 224, 224]
    """
    return transform(image)

# MediaPipe face detection and alignment
mp_face_mesh = mp.solutions.face_mesh

def detect_and_align_face(image, face_mesh, output_size=224, detection_width=640):
    """
    Detect face using MediaPipe and align to 224x224.
    Input: numpy array (BGR image), face_mesh instance
    Output: aligned PIL Image or None if no face
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Resize for detection if image is too large
    if w > detection_width:
        scale = detection_width / w
        new_h = int(h * scale)
        image_for_detection = cv2.resize(image, (detection_width, new_h))
    else:
        image_for_detection = image

    image_rgb = cv2.cvtColor(image_for_detection, cv2.COLOR_BGR2RGB)

    # Indices for alignment (same as libreface implementation)
    FACEMESH_LEFT_EYE = [(263, 249), (249, 390), (390, 373), (373, 374),
                         (374, 380), (380, 381), (381, 382), (382, 362),
                         (263, 466), (466, 388), (388, 387), (387, 386),
                         (386, 385), (385, 384), (384, 398), (398, 362)]
    FACEMESH_RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145),
                          (145, 153), (153, 154), (154, 155), (155, 133),
                          (33, 246), (246, 161), (161, 160), (160, 159),
                          (159, 158), (158, 157), (157, 173), (173, 133)]
    FACEMESH_LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                     (17, 314), (314, 405), (405, 321), (321, 375),
                     (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                     (37, 0), (0, 267), (267, 269), (269, 270), (270, 409),
                     (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
                     (87, 14), (14, 317), (317, 402), (402, 318), (318, 324),
                     (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                     (82, 13), (13, 312), (312, 311), (311, 310),
                     (310, 415), (415, 308)]

    Left_eye = []
    Right_eye = []
    Lips = []
    for (x, y) in FACEMESH_LEFT_EYE:
        if x not in Left_eye: Left_eye.append(x)
        if y not in Left_eye: Left_eye.append(y)
    for (x, y) in FACEMESH_RIGHT_EYE:
        if x not in Right_eye: Right_eye.append(x)
        if y not in Right_eye: Right_eye.append(y)
    for (x, y) in FACEMESH_LIPS:
        if x not in Lips: Lips.append(x)
        if y not in Lips: Lips.append(y)

    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    
    lm_left_eye_x = [face_landmarks.landmark[i].x for i in Left_eye]
    lm_left_eye_y = [face_landmarks.landmark[i].y for i in Left_eye]
    lm_right_eye_x = [face_landmarks.landmark[i].x for i in Right_eye]
    lm_right_eye_y = [face_landmarks.landmark[i].y for i in Right_eye]
    lm_lips_x = [face_landmarks.landmark[i].x for i in Lips]
    lm_lips_y = [face_landmarks.landmark[i].y for i in Lips]
    lm_x = lm_left_eye_x + lm_right_eye_x + lm_lips_x
    lm_y = lm_left_eye_y + lm_right_eye_y + lm_lips_y
    landmark = np.array([lm_x, lm_y]).T

    # Use original image for alignment (high quality)
    # Note: landmarks are normalized [0,1], so they work with original image size automatically
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aligned_image = image_align(Image.fromarray(original_rgb), landmark, output_size=output_size)
    return aligned_image

def extract_video_embeddings(video_path, model, device='cpu', batch_size=64):
    """
    Extract [T, 512] embeddings from video frames using batch inference.
    Input: video_path, model (loaded), device, batch_size
    Output: np.array [T, 512]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return np.array([])

    embeddings = []
    batch_frames = []
    
    # Initialize FaceMesh once for the video
    with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True,
                               max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and align directly from frame
            aligned_face = detect_and_align_face(frame, face_mesh)
            if aligned_face is None:
                continue

            # Preprocess and accumulate
            tensor_face = preprocess_face(aligned_face) # [3, 224, 224]
            batch_frames.append(tensor_face)

            # Process batch if full
            if len(batch_frames) >= batch_size:
                batch_tensor = torch.stack(batch_frames).to(device) # [B, 3, 224, 224]
                batch_emb = model.encode(batch_tensor).cpu().numpy() # [B, 512]
                embeddings.append(batch_emb)
                batch_frames = []

        # Process remaining frames
        if batch_frames:
            batch_tensor = torch.stack(batch_frames).to(device)
            batch_emb = model.encode(batch_tensor).cpu().numpy()
            embeddings.append(batch_emb)

    cap.release()
    
    if embeddings:
        return np.concatenate(embeddings, axis=0)  # [T, 512]
    else:
        return np.array([])

def extract_mosi_embeddings(mosi_root='MOSI', device='cpu'):
    """
    Extract embeddings for all MOSI clips.
    Input: mosi_root - path to MOSI dataset root
    Output: dict {video_id_clip_id: np.array [T, 512]}
    """
    # NOTE: This function imports pandas, so it might fail if pandas is broken.
    # But the user's task is to extract embeddings, maybe they can provide the list of videos differently.
    # Or we can read the csv manually.
    
    import pandas as pd
    label_path = os.path.join(mosi_root, 'label.csv')
    raw_path = os.path.join(mosi_root, 'Raw')

    df = pd.read_csv(label_path)
    embeddings_dict = {}

    model = load_fer_model(device)

    for idx, row in df.iterrows():
        video_id = row['video_id']
        clip_id = row['clip_id']
        key = f"{video_id}_{clip_id}"

        video_file = os.path.join(raw_path, video_id, f"{clip_id}.mp4")
        if not os.path.exists(video_file):
            print(f"Video not found: {video_file}")
            continue

        embeddings = extract_video_embeddings(video_file, model, device)
        embeddings_dict[key] = embeddings
        print(f"Processed {key}: {embeddings.shape}")

    return embeddings_dict