#!/usr/bin/env python3
"""
Generate embeddings for WHO disease fact sheets using Bio_ClinicalBERT.

This script:
1. Loads the Bio_ClinicalBERT model and tokenizer
2. Reads the JSONL file with medical text chunks
3. Generates embeddings using mean pooling over token embeddings
4. Saves embeddings to a new JSONL file
"""

import json
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
INPUT_FILE = Path("data/who_diseases_Full_Catalogue.jsonl")
OUTPUT_FILE = Path("data/who_embeddings.jsonl")
BATCH_SIZE = 16  # Adjust based on your GPU memory
MAX_LENGTH = 512  # Maximum sequence length for tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(token_embeddings, attention_mask):
    """
    Perform mean pooling on token embeddings, ignoring padding tokens.
    
    Args:
        token_embeddings: Tensor of shape (batch_size, seq_length, hidden_size)
        attention_mask: Tensor of shape (batch_size, seq_length) with 1s for real tokens, 0s for padding
    
    Returns:
        Tensor of shape (batch_size, hidden_size) with mean-pooled embeddings
    """
    # Expand attention mask to match embedding dimensions
    # Shape: (batch_size, seq_length) -> (batch_size, seq_length, 1)
    input_mask_expanded = attention_mask.unsqueeze(-1).float()
    
    # Sum embeddings, ignoring padding tokens
    # Shape: (batch_size, hidden_size)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    
    # Count non-padding tokens for each sequence
    # Shape: (batch_size,)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    # Compute mean by dividing sum by count
    # Shape: (batch_size, hidden_size)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings


def load_model_and_tokenizer():
    """Load Bio_ClinicalBERT model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    return model, tokenizer


def read_jsonl(file_path):
    """Read all records from a JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                records.append(json.loads(line))
    return records


def process_batch(texts, model, tokenizer):
    """
    Process a batch of texts and generate embeddings.
    
    Args:
        texts: List of text strings
        model: The Bio_ClinicalBERT model
        tokenizer: The tokenizer
    
    Returns:
        numpy array of shape (batch_size, hidden_size) with embeddings
    """
    # Tokenize the batch
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    
    # Generate token embeddings (no gradient computation for efficiency)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state shape: (batch_size, seq_length, hidden_size)
        token_embeddings = outputs.last_hidden_state
    
    # Apply mean pooling
    embeddings = mean_pooling(token_embeddings, attention_mask)
    
    # Convert to numpy and then to list of floats
    embeddings_np = embeddings.cpu().numpy()
    
    return embeddings_np


def generate_embeddings(input_file, output_file, model, tokenizer):
    """
    Generate embeddings for all records in the input JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        model: The Bio_ClinicalBERT model
        tokenizer: The tokenizer
    """
    # Read all records
    print(f"Reading records from: {input_file}")
    records = read_jsonl(input_file)
    total_records = len(records)
    print(f"Found {total_records} records")
    
    # Open output file for writing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Process in batches
        for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Processing batches"):
            batch_records = records[i:i + BATCH_SIZE]
            batch_texts = [record['text'] for record in batch_records]
            
            # Generate embeddings for this batch
            batch_embeddings = process_batch(batch_texts, model, tokenizer)
            
            # Write each record with its embedding
            for j, record in enumerate(batch_records):
                embedding = batch_embeddings[j].tolist()  # Convert to list of floats
                
                output_record = {
                    'id': record['id'],
                    'title': record['title'],
                    'text': record['text'],
                    'embedding': embedding
                }
                
                out_f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    
    print(f"\nEmbeddings saved to: {output_file}")
    print(f"Total records processed: {total_records}")


def main():
    """Main function to orchestrate the embedding generation."""
    print("=" * 60)
    print("WHO Disease Fact Sheets - Embedding Generation")
    print("=" * 60)
    
    # Check if input file exists
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Generate embeddings
    generate_embeddings(INPUT_FILE, OUTPUT_FILE, model, tokenizer)
    
    print("\n" + "=" * 60)
    print("Embedding generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

