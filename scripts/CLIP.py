import torch
import open_clip
import os, json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class ZeroShot:
    def __init__(self, 
                 project_root=proj_path,
                 description_file_name='definitions.json',
                 saved_model_file_name='clip_finetuned.pt',
                 clip_model_name = "ViT-B-16"):
                
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use device: {self.device}")

        self.description_file_path = os.path.join(project_root, "rare_animals_dataset", description_file_name)
        self.saved_model_path = os.path.join(project_root, "weights", "fine-tuned", saved_model_file_name)
        #self.saved_model_path = """C:\\Users\\a\\Desktop\\CV-rare-animal\\weights\\fine-tuned\\fin_model(1).pt"""
        print (self.saved_model_path)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=clip_model_name,
            pretrained=None,
            device=self.device
        )
        finetuned_state_dict = torch.load(self.saved_model_path, map_location=self.device)
        print (type(finetuned_state_dict))
        new_state_dict = OrderedDict() # Handle the case the weights have module preceder
        for k, v in finetuned_state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print(f"Fine-tuned model loaded from {self.saved_model_path}")

        with open(self.description_file_path, 'r', encoding='utf-8') as f:
            animal_data = json.load(f)
            self.species_names = sorted(list(animal_data.keys()))
        self.num_total_species = len(self.species_names)

        eval_text_prompts = [f"a photo of a {name}." for name in self.species_names]
        with torch.no_grad():
            text_tokens_for_eval = open_clip.tokenize(eval_text_prompts).to(self.device)
            self.eval_text_features = self.model.encode_text(text_tokens_for_eval)
            self.eval_text_features = F.normalize(self.eval_text_features, p=2, dim=-1)
        print("Text features for shot ready.") 

    def classify_image(self, image_input, top_k: int = 5):
        image = Image.fromarray(image_input[..., ::-1]).convert("RGB")
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
            image_features = F.normalize(image_features, p=2, dim=-1)

            logits = (image_features @ self.eval_text_features.T)
            top_k_actual = min(top_k, self.num_total_species)
            top_probs, top_indices = logits.topk(top_k_actual, dim=-1)

            results = []
            for i in range(top_k_actual):
                label_idx = top_indices[0, i].item()
                confidence = top_probs[0, i].item()
                label_name = self.species_names[label_idx]
                results.append({"label": label_name, "confidence": confidence})

        return results
