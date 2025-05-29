import gc
import json
import os
import sys
import warnings

import torch
import tqdm
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

# Ignore the specific FutureWarning about device argument
warnings.filterwarnings("ignore", message="The `device` argument is deprecated")

# Paths
input_file = "data/raw/descriptors_with_explainers.jsonl"
output_file = "data/processed/descriptors_with_explainers_embeddings_2.jsonl"
model_dir = "NovaSearch/stella_en_400M_v5"

# Make sure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Model configuration
vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"

# Define prompts as per official documentation
s2s_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "
s2p_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "

print("Loading Stella model...")

# Load model correctly (single instantiation)
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

# Create and load vector linear layer correctly
vector_linear = torch.nn.Linear(
    in_features=model.config.hidden_size, out_features=vector_dim
)

# Load the correct vector linear weights
try:
    vector_linear_dict = {
        k.replace("linear.", ""): v
        for k, v in torch.load(
            os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")
        ).items()
    }
    vector_linear.load_state_dict(vector_linear_dict)
    print("✅ Successfully loaded vector linear weights from model")
except Exception as e:
    print(f"❌ Error loading vector linear weights: {e}")
    print("This will result in poor embedding quality!")
    sys.exit(1)

vector_linear.cuda()

print("Model loaded successfully!")


def embed_text(texts, use_prompt=True, prompt_type="s2s"):
    """
    Embed text using Stella model following official implementation

    Args:
        texts: List of texts to embed
        use_prompt: Whether to use prompts (recommended for best performance)
        prompt_type: "s2s" for similarity tasks, "s2p" for retrieval tasks

    Returns:
        Normalized embeddings as numpy array
    """
    # Apply prompts if requested
    if use_prompt:
        if prompt_type == "s2s":
            processed_texts = [s2s_prompt + text for text in texts]
        elif prompt_type == "s2p":
            processed_texts = [s2p_prompt + text for text in texts]
        else:
            raise ValueError("prompt_type must be 's2s' or 's2p'")
    else:
        processed_texts = texts

    with torch.no_grad():
        # Tokenize with official settings
        input_data = tokenizer(
            processed_texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to GPU
        input_data = {k: v.cuda() for k, v in input_data.items()}

        # Get model outputs
        attention_mask = input_data["attention_mask"]
        last_hidden_state = model(**input_data)[0]

        # Apply attention masking and mean pooling
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        # Apply vector linear transformation
        vectors = vector_linear(vectors)

        # Move to CPU and normalize
        vectors = vectors.cpu().numpy()
        vectors = normalize(vectors)

    return vectors


def test_embedding_consistency():
    """Test that batch embedding produces same results as individual embedding"""
    print("Running embedding consistency test...")

    # Test inputs - using descriptive texts similar to your use case
    test_texts = [
        "machine learning algorithm",
        "natural language processing",
        "computer vision model",
    ]

    # Test both with and without prompts
    for use_prompt in [True, False]:
        for prompt_type in ["s2s", "s2p"] if use_prompt else [None]:
            prompt_desc = (
                f" with {prompt_type} prompt" if use_prompt else " without prompts"
            )
            print(f"  Testing{prompt_desc}...")

            # Method 1: Batch embedding
            if use_prompt:
                batch_embeddings = embed_text(
                    test_texts, use_prompt=True, prompt_type=prompt_type
                )
            else:
                batch_embeddings = embed_text(test_texts, use_prompt=False)

            # Method 2: Individual embedding
            individual_embeddings = []
            for text in test_texts:
                if use_prompt:
                    embedding = embed_text(
                        [text], use_prompt=True, prompt_type=prompt_type
                    )
                else:
                    embedding = embed_text([text], use_prompt=False)
                individual_embeddings.append(embedding[0])

            # Convert to numpy array for comparison
            individual_embeddings = torch.tensor(individual_embeddings).numpy()

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
                    print(f"    ❌ FAILED for text {i}: '{test_texts[i]}'")
                    print(f"       Max difference: {diff}")
                    return False

            print(f"    ✅ PASSED (max diff: {max_diff:.2e})")

    print("✅ All embedding consistency tests PASSED")
    return True


def verify_against_official_example():
    """Verify our implementation matches the official example output format"""
    print("Verifying against official documentation example...")

    # Use exact examples from the documentation
    test_queries = [
        "What are some ways to reduce stress?",
        "What are the benefits of drinking green tea?",
    ]

    test_docs = [
        "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity.",
        "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body.",
    ]

    # Embed queries with s2p prompt (for retrieval)
    query_embeddings = embed_text(test_queries, use_prompt=True, prompt_type="s2p")

    # Embed docs without prompts (as per documentation)
    doc_embeddings = embed_text(test_docs, use_prompt=False)

    # Compute similarities
    similarities = query_embeddings @ doc_embeddings.T

    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Doc embeddings shape: {doc_embeddings.shape}")
    print("Similarities matrix:")
    print(similarities)

    # Check if we get reasonable similarity values (should be positive and < 1)
    if similarities.min() > 0 and similarities.max() < 1:
        print("✅ Similarity values look reasonable")
        return True
    else:
        print("❌ Similarity values look suspicious")
        return False


# Run consistency tests
print("\n" + "=" * 50)
print("RUNNING EMBEDDING TESTS")
print("=" * 50)

if not test_embedding_consistency():
    print("❌ Embedding consistency test failed. Exiting.")
    sys.exit(1)

if not verify_against_official_example():
    print("❌ Official example verification failed. Results may be unreliable.")
    # Don't exit here, just warn

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
print(
    f"Using no prompts for document embedding (descriptors will be stored/retrieved later)"
)

with open(output_file, "w") as out_f:
    for i in tqdm.tqdm(range(0, len(unique_descriptors_list), batch_size)):
        batch = unique_descriptors_list[i : i + batch_size]

        try:
            # Embed batch without prompts (document embedding)
            embeddings = embed_text(batch, use_prompt=False)

            # Write results to output file
            for j, desc in enumerate(batch):
                output_data = {
                    "descriptor": desc,
                    "embedding": embeddings[j].tolist(),
                    "embedding_dim": vector_dim,
                    "model": model_dir,
                    "prompt_type": None,  # No prompts for document embedding
                }
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

print(f"Done! Embedded {len(unique_descriptors)} unique descriptors with Stella model.")
print(f"Output saved to: {output_file}")
