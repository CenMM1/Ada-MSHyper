import torch
import numpy as np
import pandas as pd
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
import librosa
from fer_extractor import extract_video_embeddings as extract_visual_embeddings, load_fer_model

# Load models globally to avoid reloading
wav2vec_processor = None
wav2vec_model = None
bert_tokenizer = None
bert_model = None

def load_audio_model(device='cpu'):
    global wav2vec_processor, wav2vec_model
    if wav2vec_processor is None:
        wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    return wav2vec_processor, wav2vec_model

def load_text_model(device='cpu'):
    global bert_tokenizer, bert_model
    if bert_tokenizer is None:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    return bert_tokenizer, bert_model

def extract_audio_features(audio_path, device='cpu'):
    """
    Extract audio features using Wav2Vec2.
    Input: audio_path (wav file)
    Output: np.array [seq_len, 768]
    """
    processor, model = load_audio_model(device)
    model.eval()
    with torch.no_grad():
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)  # Wav2Vec expects 16kHz
        # Process
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        # Use last hidden state as features
        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [seq_len, 768]
    return features

def extract_text_features(text, device='cpu'):
    """
    Extract text features using BERT.
    Input: text (string)
    Output: np.array [768]
    """
    tokenizer, model = load_text_model(device)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        # Use [CLS] token embedding
        features = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # [768]
    return features

def extract_multimodal_features(mosi_root='mosei', device='cpu'):
    """
    Extract multimodal features (visual, audio, text) for MOSEI dataset.
    Output: dict {video_id_clip_id: {'visual': np.array [T, 512], 'audio': np.array [seq_len, 768], 'text': np.array [768]}}
    """
    import pandas as pd
    label_path = os.path.join(mosi_root, 'label.csv')
    raw_path = os.path.join(mosi_root, 'Raw')

    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return {}

    df = pd.read_csv(label_path)
    features_dict = {}

    # Load visual model
    visual_model = load_fer_model(device)

    for idx, row in df.iterrows():
        video_id = row['video_id']
        clip_id = row['clip_id']
        text = row.get('text', '')  # Assuming 'text' column exists, default empty
        key = f"{video_id}_{clip_id}"

        # Visual features
        video_file = os.path.join(raw_path, video_id, f"{clip_id}.mp4")
        if os.path.exists(video_file):
            visual_emb = extract_visual_embeddings(video_file, visual_model, device)
        else:
            visual_emb = np.array([])
            print(f"Video not found: {video_file}")

        # Audio features (assume audio file exists, e.g., clip_id.wav)
        audio_file = os.path.join(raw_path, video_id, f"{clip_id}.wav")
        if os.path.exists(audio_file):
            audio_emb = extract_audio_features(audio_file, device)
        else:
            audio_emb = np.array([])
            print(f"Audio not found: {audio_file}")

        # Text features
        text_emb = extract_text_features(text, device)

        features_dict[key] = {
            'visual': visual_emb,
            'audio': audio_emb,
            'text': text_emb
        }
        print(f"Processed {key}")

    return features_dict

if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = extract_multimodal_features('mosi', device)
    print("Extraction complete")