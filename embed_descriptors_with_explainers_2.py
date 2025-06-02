import gc
import json
import os
import sys
import warnings

import torch
import tqdm
from sentence_transformers import SentenceTransformer

# Ignore warnings
warnings.filterwarnings("ignore")

# Paths
input_file = "data/raw/descriptors_with_explainers.jsonl"
output_file = "data/processed/descriptors_with_explainers_embeddings_3.jsonl"
model_name = "dunzhang/stella_en_400M_v5"

# Make sure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("Loading Stella model with SentenceTransformers...")

# Load model using SentenceTransformers
model = SentenceTransformer(
    model_name,
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)
model.eval()

print("Model loaded successfully!")


def embed_text(texts):
    """
    Embed text using Stella model via SentenceTransformers
    """
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            # convert_to_tensor=False,  # Return numpy arrays
            # normalize_embeddings=True,  # Normalize automatically
            batch_size=32,
        )
    return embeddings


def test_embedding_consistency():
    """Test that batch embedding produces same results as individual embedding"""
    print("Running embedding consistency test...")

    # Test inputs
    test_texts = [
        "machine learning algorithm",
        "natural language processing model",
        "computer vision model model model model model model model model model model v v v model v v model model",
    ]

    # Method 1: Batch embedding
    batch_embeddings = embed_text(test_texts)

    # Method 2: Individual embedding
    individual_embeddings = []
    for text in test_texts:
        embedding = embed_text([text])
        individual_embeddings.append(embedding[0])

    # Compare results
    tolerance = 1e-4
    max_diff = 0
    for i, (batch_emb, indiv_emb) in enumerate(
        zip(batch_embeddings, individual_embeddings)
    ):
        diff = abs(batch_emb - indiv_emb).max()
        max_diff = max(max_diff, diff)
        if not torch.allclose(
            torch.tensor(batch_emb), torch.tensor(indiv_emb), atol=tolerance
        ):
            print(f"❌ FAILED for text {i}: '{test_texts[i]}'")
            print(f"   Max difference: {diff}")
            return False

    print(f"✅ PASSED (max diff: {max_diff:.2e})")
    return True


# Run tests
print("\n" + "=" * 50)
print("RUNNING EMBEDDING TESTS")
print("=" * 50)

if not test_embedding_consistency():
    print("❌ Embedding consistency test failed. Exiting.")
    sys.exit(1)

print("\n" + "=" * 50)
print("STARTING MAIN PROCESSING")
print("=" * 50)

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

                # Process each descriptor (lowercase and strip, but keep the full text)
                for descriptor in best_descriptors:
                    # Process the descriptor (lowercase and strip)
                    processed_descriptor = descriptor.lower().strip()
                    if processed_descriptor:
                        unique_descriptors.add(processed_descriptor)
        except Exception as e:
            print(f"Error processing line {line_idx}: {e}")
            continue

        # Print progress periodically
        if line_idx % 10000 == 0 and line_idx > 0:
            print(
                f"Processed {line_idx} lines, found {len(unique_descriptors)} unique descriptors so far"
            )

print(f"Found {len(unique_descriptors)} unique descriptors from {total_docs} documents")

# Convert set to list and sort alphabetically
unique_descriptors_list = sorted(list(unique_descriptors))
print(
    f"Sorting completed. Embedding {len(unique_descriptors_list)} unique descriptors..."
)

# Now process in batches and write to output file
print("Embedding unique descriptors and writing to output file...")
print("Using SentenceTransformers (no manual prompt handling needed)")

with open(output_file, "w") as out_f:
    for i in tqdm.tqdm(range(0, len(unique_descriptors_list), batch_size)):
        batch = unique_descriptors_list[i : i + batch_size]

        try:
            # Embed batch
            embeddings = embed_text(batch)

            # Write results to output file
            for j, desc in enumerate(batch):
                output_data = {
                    "descriptor": desc,
                    "embedding": embeddings[j].tolist(),
                }
                out_f.write(json.dumps(output_data) + "\n")

            # Periodic cleanup to prevent memory buildup
            if i % (batch_size * 50) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error embedding batch starting at index {i}: {e}")
            continue

print(f"Done! Embedded {len(unique_descriptors)} unique descriptors with Stella model.")
print(f"Output saved to: {output_file}")
