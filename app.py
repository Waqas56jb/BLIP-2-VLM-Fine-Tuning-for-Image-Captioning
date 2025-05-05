import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torchvision import transforms
import evaluate
from tqdm import tqdm
import uuid
import json
from sklearn.model_selection import train_test_split

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Custom Dataset class
class CustomImageCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, processor, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.data = self._load_captions(captions_file)
        self.subsample_size = 5000  # Subsample to ≤ 5K pairs

    def _load_captions(self, captions_file):
        data = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    image_name, caption = line.strip().split('|', 1)
                    if image_name in data:
                        data[image_name].append(caption)
                    else:
                        data[image_name] = [caption]
        # Subsample to 5000 pairs
        sampled_data = {}
        pair_count = 0
        for image_name, captions in list(data.items())[:self.subsample_size // 5]:  # ~5 captions per image
            if pair_count < self.subsample_size:
                sampled_data[image_name] = captions[:min(5, len(captions))]
                pair_count += len(captions)
            if pair_count >= self.subsample_size:
                break
        return sampled_data

    def __len__(self):
        return sum(min(5, len(captions)) for captions in self.data.values())

    def __getitem__(self):
        # Randomly select an image and one of its captions
        image_name = np.random.choice(list(self.data.keys()))
        caption = np.random.choice(self.data[image_name])
        image_path = os.path.join(self.image_dir, image_name)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze(),
            'pixel_values': encoding.pixel_values.squeeze(),
            'caption': caption
        }

def compute_metrics(references, predictions):
    """Compute BLEU-4, METEOR, ROUGE-L, SPICE, Self-BLEU, Distinct-n"""
    bleu_scores = []
    meteor_scores = []
    rouge_l_scores = []
    spice = evaluate.load("spice")
    distinct_n = set()
    self_bleu_scores = []
    smoothie = SmoothingFunction().method1
    
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    for i, (refs, pred) in enumerate(zip(references, predictions)):
        ref_tokens = [nltk.word_tokenize(ref) for ref in refs]
        pred_tokens = nltk.word_tokenize(pred)
        
        # BLEU-4
        bleu = corpus_bleu([[ref] for ref in ref_tokens], [pred_tokens], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu_scores.append(bleu)
        
        # METEOR
        meteor = meteor_score(ref_tokens, pred_tokens)
        meteor_scores.append(meteor)
        
        # ROUGE-L
        rouge_score = rouge.score(' '.join(refs[0]), pred)['rougeL'].f1
        rouge_l_scores.append(rouge_score)
        
        # SPICE (requires reference and prediction as strings)
        spice_score = spice.compute(predictions=[pred], references=[refs[0]])['f1']
        
        # Distinct-n (Distinct-1)
        distinct_n.update(pred_tokens)
        
        # Self-BLEU (compare with other predictions)
        if i > 0:
            self_bleu = sentence_bleu([nltk.word_tokenize(predictions[j]) for j in range(i) if j != i], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            self_bleu_scores.append(self_bleu)
    
    distinct_1 = len(distinct_n) / sum(len(nltk.word_tokenize(pred)) for pred in predictions) if predictions else 0
    self_bleu_mean = np.mean(self_bleu_scores) if self_bleu_scores else 0
    
    return {
        'BLEU-4': np.mean(bleu_scores) * 100,
        'METEOR': np.mean(meteor_scores) * 100,
        'ROUGE-L': np.mean(rouge_l_scores) * 100,
        'SPICE': spice_score * 100,
        'Self-BLEU': self_bleu_mean * 100,
        'Distinct-1': distinct_1 * 100
    }

def generate_description(model, processor, image, method='beam', **kwargs):
    """Implement beam search, top-k, top-p sampling with temperature control"""
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    if method == 'beam':
        outputs = model.generate(
            **inputs,
            num_beams=kwargs.get('num_beams', 5),
            max_length=128,
            length_penalty=1.0,
            early_stopping=True
        )
    elif method == 'top_k':
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_length=128,
            top_k=kwargs.get('top_k', 50),
            temperature=kwargs.get('temperature', 1.0)
        )
    elif method == 'top_p':
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_length=128,
            top_p=kwargs.get('top_p', 0.9),
            temperature=kwargs.get('temperature', 1.0)
        )
    
    return processor.decode(outputs[0], skip_special_tokens=True)

def train_model(model, processor, train_loader, val_loader, epochs=3, lr=5e-5, strategy='full'):
    """Fine-tune with full, prompt-tuning, or layer-freeze strategies"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Apply fine-tuning strategy
    if strategy == 'prompt_tuning':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.text_decoder.parameters():
            param.requires_grad = True
    elif strategy == 'layer_freeze':
        for param in model.vision_model.parameters():
            param.requires_grad = False
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items() if k != 'caption'}
            labels = batch['input_ids']
            
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=labels,
                attention_mask=batch['attention_mask'],
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_references = []
        val_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if k != 'caption'}
                images = batch['pixel_values']
                refs = [item['caption'] for item in batch]
                val_references.extend([[ref] for ref in refs])  # List of lists for multiple references
                
                for img in images:
                    pred = generate_description(model, processor, img, method='beam')
                    val_predictions.append(pred)
        
        metrics = compute_metrics(val_references, val_predictions)
        print(f"Epoch {epoch+1} Metrics ({strategy}):", metrics)
        
        model.save_pretrained(f"blip2_finetuned_{strategy}_epoch_{epoch+1}")
        processor.save_pretrained(f"blip2_finetuned_{strategy}_epoch_{epoch+1}")

def qualitative_analysis(model, processor, test_dataset, num_samples=20):
    """Analyze hallucinations, repetitions, omissions"""
    analysis_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for _ in range(num_samples):
        item = test_dataset[np.random.randint(len(test_dataset))]
        image = Image.open(os.path.join('images', np.random.choice(list(dataset.data.keys())))).convert('RGB')
        reference = item['caption']
        
        beam_desc = generate_description(model, processor, image, method='beam', num_beams=5)
        top_k_desc = generate_description(model, processor, image, method='top_k', top_k=50, temperature=0.7)
        top_p_desc = generate_description(model, processor, image, method='top_p', top_p=0.9, temperature=0.7)
        
        analysis = {
            'image_id': str(uuid.uuid4()),
            'reference': reference,
            'beam_description': beam_desc,
            'top_k_description': top_k_desc,
            'top_p_description': top_p_desc,
            'hallucinations': [],
            'repetitions': [],
            'omissions': []
        }
        
        ref_tokens = set(nltk.word_tokenize(reference.lower()))
        beam_tokens = nltk.word_tokenize(beam_desc.lower())
        
        # Hallucinations
        for token in beam_tokens:
            if token not in ref_tokens and token not in ['a', 'an', 'the', 'is', 'are', 'of', 'in', 'and']:
                analysis['hallucinations'].append(token)
        
        # Repetitions
        seen = set()
        for token in beam_tokens:
            if token in seen:
                analysis['repetitions'].append(token)
            seen.add(token)
        
        # Omissions
        for token in ref_tokens:
            if token not in beam_tokens and token not in ['a', 'an', 'the', 'is', 'are', 'of', 'in', 'and']:
                analysis['omissions'].append(token)
        
        analysis_results.append(analysis)
    
    with open('qualitative_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    return analysis_results

def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load processor and model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Load custom dataset
    dataset = CustomImageCaptionDataset(
        image_dir='images',
        captions_file='captions.txt',
        processor=processor,
        transform=transform
    )
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)
    
    # Train with different strategies
    for strategy in ['full', 'prompt_tuning', 'layer_freeze']:
        train_model(model, processor, train_loader, val_loader, epochs=3, strategy=strategy)
    
    # Evaluate on test set
    model.eval()
    test_references = []
    test_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items() if k != 'caption'}
            images = batch['pixel_values']
            refs = [item['caption'] for item in batch]
            test_references.extend([[ref] for ref in refs])
            
            for img in images:
                pred = generate_description(model, processor, img, method='beam')
                test_predictions.append(pred)
    
    final_metrics = compute_metrics(test_references, test_predictions)
    print("Final Test Metrics:", final_metrics)
    
    # Qualitative analysis
    analysis_results = qualitative_analysis(model, processor, test_dataset)
    
    # Save results
    with open('final_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()