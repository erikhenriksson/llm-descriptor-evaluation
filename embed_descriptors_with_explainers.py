import gc
import json
import os

import torch
import tqdm
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

# Set GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
input_file = "data/raw/descriptors_with_explainers.jsonl"
output_file = "data/processed/descriptors_with_explainers_embeddings.jsonl"
model_dir = "NovaSearch/stella_en_400M_v5"  # Using Hugging Face model directly

# Make sure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load model
vector_dim = 1024
model = (
    AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_memory_efficient_attention=False,
        unpad_inputs=False,
    )
    .cuda()
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Create vector linear layer
vector_linear = torch.nn.Linear(
    in_features=model.config.hidden_size, out_features=vector_dim
)

# Try to load the vector linear weights from different possible locations
try:
    # Check if the model has the vector linear weights embedded
    if hasattr(model, "vector_linear"):
        print("Using model's built-in vector_linear layer")
        vector_linear = model.vector_linear
    else:
        # Try to find weights in the model's state dict
        vector_linear_dict = {
            k.replace("vector_linear.", ""): v
            for k, v in {
                k: v for k, v in model.state_dict().items() if "vector_linear" in k
            }.items()
        }

        if vector_linear_dict:
            print("Loading vector_linear from model's state dict")
            vector_linear.load_state_dict(vector_linear_dict)
        else:
            print(
                "Warning: Vector linear layer not found, using default initialization"
            )
except Exception as e:
    print(f"Error loading vector linear layer: {e}")
    print("Using default initialization")

# Move vector linear to device
vector_linear = vector_linear.to(device)


# Function to embed text
def embed_text(texts):
    with torch.no_grad():
        input_data = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_data = {k: v.to(device) for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        vectors = vector_linear(vectors)
        # Move to CPU for numpy conversion
        vectors = vectors.cpu().numpy()
        # Normalize
        vectors = normalize(vectors)
    return vectors


# Batch size for embedding
batch_size = 32

# First, collect all unique descriptors
print("Collecting unique descriptors from input file...")
unique_descriptors = set()
total_docs = 0

with open(input_file, "r") as in_f:
    for line_idx, line in enumerate(tqdm.tqdm(in_f)):
        try:
            doc_data = json.loads(line)
            total_docs += 1

            # Find the index with the highest similarity
            similarities = doc_data.get("similarity", [0])
            if not similarities:
                continue

            best_index = similarities.index(max(similarities))

            # Get the best general descriptors list
            general_lists = doc_data.get("general", [])
            if best_index < len(general_lists) and general_lists[best_index]:
                best_descriptors = general_lists[best_index]

                # Process each descriptor (lowercase and strip)
                for descriptor_with_explanation in best_descriptors:
                    # Extract just the descriptor part (before the colon)
                    descriptor_parts = descriptor_with_explanation.split(":", 1)
                    if len(descriptor_parts) > 0:
                        descriptor = descriptor_parts[0].lower().strip()
                        if descriptor:
                            unique_descriptors.add(descriptor)
        except Exception as e:
            print(f"Error processing line {line_idx}: {e}")
            continue

        # Print progress periodically
        if line_idx % 10000 == 0 and line_idx > 0:
            print(
                f"Processed {line_idx} lines, found {len(unique_descriptors)} unique descriptors so far"
            )

print(f"Found {len(unique_descriptors)} unique descriptors from {total_docs} documents")

# Convert set to list for batch processing
unique_descriptors_list = list(unique_descriptors)

# Now process in batches and write to output file
print("Embedding unique descriptors and writing to output file...")
with open(output_file, "w") as out_f:
    for i in tqdm.tqdm(range(0, len(unique_descriptors_list), batch_size)):
        batch = unique_descriptors_list[i : i + batch_size]

        try:
            # Embed batch
            embeddings = embed_text(batch)

            # Write results to output file
            for j, desc in enumerate(batch):
                output_data = {"descriptor": desc, "embedding": embeddings[j].tolist()}
                out_f.write(json.dumps(output_data) + "\n")

            # Periodic cleanup to prevent memory buildup
            if i % (batch_size * 50) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error embedding batch starting at index {i}: {e}")
            # Continue with next batch
            continue

print(f"Done! Embedded {len(unique_descriptors)} unique descriptors.")
