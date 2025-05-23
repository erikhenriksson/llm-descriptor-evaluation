import json
import os

# Set Hugging Face cache directory
os.environ["HF_HOME"] = ".hf_cache"

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict


# Path to input files
data_dir = "data"
descriptors_file = os.path.join(
    data_dir, "final_zero_vocab/descriptors_final_zero_vocab.jsonl"
)
edu_ids_file = os.path.join(data_dir, "edu_ids.jsonl")
output_file_base = os.path.join(data_dir, "clustered_descriptors")

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Path to save/load embeddings
embeddings_file = os.path.join(data_dir, "descriptor_embeddings.npy")
descriptors_list_file = os.path.join(data_dir, "descriptor_list.json")

# Load educational IDs
edu_ids = set()
with open(edu_ids_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            # Try to get the ID - assuming it might be directly the ID or in an 'id' field
            if isinstance(data, dict):
                doc_id = data.get("doc_id")
                if doc_id:
                    edu_ids.add(doc_id)
            elif isinstance(data, str):
                edu_ids.add(data)
            elif isinstance(data, int):
                edu_ids.add(str(data))
        except json.JSONDecodeError:
            # If not JSON, try to use the line directly as ID
            line = line.strip()
            if line:
                edu_ids.add(line)

print(f"Loaded {len(edu_ids)} educational document IDs")

# Load descriptors and documents
docs_data = []
descriptors_counter = defaultdict(int)

# Count total lines in the file for tqdm
total_lines = sum(1 for _ in open(descriptors_file, "r"))
print(f"Found {total_lines} lines in {descriptors_file}")

print("Loading documents and descriptors...")
with open(descriptors_file, "r") as f:
    for line in tqdm(f, total=total_lines, desc="Processing documents"):
        data = json.loads(line)
        doc_id = data.get("doc_id")

        # Extract descriptors from the 'general' key
        descriptors = data.get("general", [])

        # Make sure descriptors is a flat list of strings
        if descriptors and isinstance(descriptors, list):
            # If descriptors is a list of lists, flatten it
            flat_descriptors = []
            for desc in descriptors:
                if isinstance(desc, list):
                    flat_descriptors.extend(desc)
                else:
                    flat_descriptors.append(desc)

            # Add only hashable (string) descriptors to the set, convert to lowercase
            string_descriptors = [
                str(d).lower().strip() for d in flat_descriptors if d is not None
            ]

            # Count occurrences of each descriptor
            for descriptor in string_descriptors:
                descriptors_counter[descriptor] += 1

            docs_data.append(
                {
                    "doc_id": doc_id,
                    "descriptors": string_descriptors,
                    "is_educational": doc_id in edu_ids,
                }
            )

# Print descriptor statistics
total_descriptors = sum(descriptors_counter.values())
unique_descriptors = len(descriptors_counter)
print(
    f"Found {total_descriptors} total descriptors with {unique_descriptors} unique values"
)

# Print most common descriptors
print("\nTop 10 most common descriptors:")
for descriptor, count in sorted(
    descriptors_counter.items(), key=lambda x: x[1], reverse=True
)[:10]:
    print(
        f"  - '{descriptor}': {count} occurrences ({count / total_lines:.1%} of documents)"
    )

# Option to filter descriptors by frequency
min_frequency = 5  # Can be adjusted based on your specific needs
max_descriptors = 50000  # Limit to prevent OOM errors
print(f"\nFiltering descriptors to those appearing at least {min_frequency} times...")

filtered_descriptors = {
    desc for desc, count in descriptors_counter.items() if count >= min_frequency
}
if len(filtered_descriptors) > max_descriptors:
    print(
        f"WARNING: Still too many descriptors ({len(filtered_descriptors)}). Taking top {max_descriptors} by frequency."
    )
    filtered_descriptors = {
        desc
        for desc, count in sorted(
            descriptors_counter.items(), key=lambda x: x[1], reverse=True
        )[:max_descriptors]
    }

descriptors_set = filtered_descriptors
print(f"After filtering: {len(descriptors_set)} unique descriptors to be clustered")

print(
    f"Loaded {len(docs_data)} documents with {len(descriptors_set)} unique descriptors"
)

# Load the embedding model
print("Loading embedding model...")
model_dir = "Marqo/dunzhang-stella_en_400M_v5"
# Suppress deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Check CUDA availability and print clear status
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")

model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Move model to GPU explicitly
model = model.to(device)
print(f"Model is on device: {next(model.parameters()).device}")
model.eval()

# Path to save/load embeddings
embeddings_file = "descriptor_embeddings.npy"
descriptors_list_file = "descriptor_list.json"

# Check if pre-computed embeddings exist
if os.path.exists(embeddings_file) and os.path.exists(descriptors_list_file):
    print(f"Loading pre-computed embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)
    with open(descriptors_list_file, "r") as f:
        loaded_descriptors = json.load(f)

    # Verify the descriptors match our current set
    if set(loaded_descriptors) == descriptors_set:
        print("Loaded embeddings match current descriptors set")
        descriptors_list = loaded_descriptors
    else:
        print("Loaded descriptors don't match current set. Recomputing embeddings...")
        compute_embeddings = True
else:
    print("No pre-computed embeddings found. Computing embeddings...")
    compute_embeddings = True

if "compute_embeddings" in locals() and compute_embeddings:
    # Embed all unique descriptors
    print("Embedding descriptors...")
    descriptors_list = list(descriptors_set)
    embeddings = []

    # Process in batches to avoid memory issues
    batch_size = 32
    total_batches = (len(descriptors_list) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(descriptors_list), batch_size),
        desc="Computing embeddings",
        total=total_batches,
    ):
        batch = descriptors_list[i : i + batch_size]
        with torch.no_grad():
            input_data = tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            # Explicitly move input data to device
            input_data = {k: v.to(device) for k, v in input_data.items()}

            # Double-check input is on correct device
            if device.type == "cuda" and not input_data["input_ids"].is_cuda:
                print("WARNING: Input was not moved to CUDA!")
                input_data = {k: v.to(device) for k, v in input_data.items()}

            attention_mask = input_data["attention_mask"]
            last_hidden_state = model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            batch_vectors = (
                last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            )
            batch_vectors = normalize(batch_vectors.cpu().numpy())
            embeddings.append(batch_vectors)

    # Combine all batches
    embeddings = np.vstack(embeddings)

    # Save embeddings and descriptor list for future use
    print(f"Saving embeddings to {embeddings_file}")
    np.save(embeddings_file, embeddings)
    with open(descriptors_list_file, "w") as f:
        json.dump(descriptors_list, f)


# Function to cluster descriptors at different thresholds and map original descriptors to clusters
def cluster_descriptors(descriptors_list, embeddings, threshold):
    print(f"Running AgglomerativeClustering with threshold {threshold}...")
    # Create clustering model
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )

    # Perform clustering
    cluster_labels = clustering.fit_predict(embeddings)

    # Create mapping from original descriptor to cluster
    descriptor_to_cluster = {}
    clusters = defaultdict(list)
    cluster_indices = defaultdict(list)

    for i, (descriptor, label) in enumerate(zip(descriptors_list, cluster_labels)):
        descriptor_to_cluster[descriptor] = label
        clusters[label].append(descriptor)
        cluster_indices[label].append(i)

    # For each cluster, choose a representative based on centrality
    cluster_representatives = {}

    for cluster_id, descriptors in tqdm(
        clusters.items(), desc="Selecting cluster representatives"
    ):
        if len(descriptors) == 1:
            # If only one descriptor in cluster, use it as representative
            cluster_representatives[cluster_id] = descriptors[0]
        else:
            # Get indices and embeddings for this cluster
            indices = cluster_indices[cluster_id]
            cluster_embeddings = embeddings[indices]

            # Compute centroid of the cluster
            centroid = np.mean(cluster_embeddings, axis=0)

            # Find descriptor closest to centroid (most central)
            distances = []
            for i, idx in enumerate(indices):
                # Compute cosine similarity (1 - cosine distance)
                similarity = np.dot(centroid, embeddings[idx]) / (
                    np.linalg.norm(centroid) * np.linalg.norm(embeddings[idx])
                )
                distances.append((1 - similarity, i))

            # Get the most central descriptor (lowest distance to centroid)
            most_central_idx = indices[min(distances, key=lambda x: x[0])[1]]
            representative = descriptors_list[most_central_idx]

            # If the most central descriptor is very long (>30 chars),
            # also consider shorter descriptors that are still close to the centroid
            if len(representative) > 30:
                # Sort by distance to centroid
                sorted_distances = sorted(distances, key=lambda x: x[0])

                # Look for a shorter descriptor among the top 3 most central (if cluster has at least 3)
                for dist, idx_in_cluster in sorted_distances[
                    : min(3, len(sorted_distances))
                ]:
                    candidate = descriptors_list[indices[idx_in_cluster]]
                    # If this descriptor is at least 30% shorter and still close to centroid
                    if len(candidate) < 0.7 * len(representative):
                        representative = candidate
                        break

            cluster_representatives[cluster_id] = representative

    return descriptor_to_cluster, cluster_representatives


# Write base case without clustering
print("\nSaving base case (no clustering)...")
base_output_file = f"{output_file_base}_base.jsonl"
print(f"Writing original descriptors to {base_output_file}...")
with open(base_output_file, "w") as f:
    for doc in tqdm(docs_data, desc="Processing documents (base case)"):
        # Use original descriptors directly without clustering
        output_data = {
            "doc_id": doc["doc_id"],
            "general_base": doc["descriptors"],  # Use direct descriptors
            "is_educational": doc["is_educational"],
        }
        f.write(json.dumps(output_data) + "\n")

print(f"Base case: Saved {len(docs_data)} documents with original descriptors")


# Test different thresholds
thresholds = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

for threshold in thresholds:
    print(f"\nClustering with threshold {threshold}...")
    descriptor_to_cluster, cluster_representatives = cluster_descriptors(
        descriptors_list, embeddings, threshold
    )

    # Update documents with clustered descriptors
    output_key = f"general_{threshold}"
    output_file = f"{output_file_base}_{threshold}.jsonl"

    print(f"Writing results to {output_file}...")
    with open(output_file, "w") as f:
        for doc in tqdm(
            docs_data, desc=f"Processing documents (threshold={threshold})"
        ):
            # Map original descriptors to cluster representatives
            original_descriptors = doc["descriptors"]
            clustered_descriptors = []

            for desc in original_descriptors:
                if (
                    desc in descriptor_to_cluster
                ):  # Make sure descriptor exists in our mapping
                    cluster_id = descriptor_to_cluster[desc]
                    representative = cluster_representatives[cluster_id]
                    if representative not in clustered_descriptors:
                        clustered_descriptors.append(representative)

            # Write to output file
            output_data = {
                "doc_id": doc["doc_id"],
                output_key: clustered_descriptors,
                "is_educational": doc["is_educational"],
            }
            f.write(json.dumps(output_data) + "\n")

    # Count clusters
    num_clusters = len(set(descriptor_to_cluster.values()))
    print(
        f"Threshold {threshold}: Created {num_clusters} clusters from {len(descriptors_set)} descriptors"
    )

print("\nProcessing complete!")
