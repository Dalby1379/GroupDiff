# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

"""
Script to create FAISS index from ImageNet features metadata.

This script loads features specified in a metadata.json file (pointing to .npy files),
aggregates them, and builds a FAISS index for similarity search.
"""

import argparse
import os
import json
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import faiss
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")


def load_npy_file(file_info: tuple[str, str]) -> np.ndarray | None:
    """Helper to load a single npy file."""
    base_path, rel_path = file_info
    full_path = os.path.join(base_path, rel_path)
    try:
        return np.load(full_path)
    except Exception:
        # logger.warning("Failed to load %s: %s", full_path, e)
        return None


def create_faiss_index(
    metadata_path: str,
    feature_key: str,
    output_dir: str,
    index_name: str = "imagenet_index.faiss",
    num_workers: int = 32,
    batch_size: int = 10000,
    index_type: str = "flat",
    nlist: int = 100,
    pq_m: int = 8,
    pq_bits: int = 8,
    nprobe: int = 10,
) -> None:
    """
    Load features from metadata and create FAISS index.

    Args:
        metadata_path (str): Path to metadata.json
        feature_key (str): Key in 'features' dict to use (e.g., 'dinov2-l', 'clip-b')
        output_dir (str): Directory to save the index
        index_name (str): Name of the output index file
        num_workers (int): Number of threads for parallel loading
        index_type (str): Type of FAISS index ("flat", "ivf", "hnsw", "ivfpq")
        nlist (int): Number of clusters for IVF index
        pq_m (int): Number of sub-vectors for Product Quantization
        pq_bits (int): Number of bits per sub-vector for PQ
        nprobe (int): Number of clusters to search during query
    """
    if not FAISS_AVAILABLE or faiss is None:
        logger.error("FAISS is not installed. Cannot create index.")
        return

    start_time = time.time()

    # 1. Load Metadata
    logger.info("Loading metadata from: %s", metadata_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    if "samples" not in metadata:
        raise ValueError("Metadata JSON must contain 'samples' key")

    samples = metadata["samples"]
    num_samples = len(samples)
    logger.info("Found %d samples in metadata", num_samples)

    if num_samples == 0:
        logger.warning("No samples found. Exiting.")
        return

    # 2. Prepare file paths
    # Assuming feature files are relative to the metadata file's directory
    base_dir = os.path.dirname(metadata_path)

    file_paths = []
    valid_samples_count = 0

    for s in samples:
        if "features" in s and feature_key in s["features"]:
            file_paths.append((base_dir, s["features"][feature_key]))
            valid_samples_count += 1
        else:
            # Handle missing features if necessary, or skip
            pass

    if valid_samples_count == 0:
        logger.error("No samples found with feature key '%s'. Available keys might differ.", feature_key)
        return

    logger.info("Preparing to load %d feature files for key '%s'...", valid_samples_count, feature_key)

    # 3. Load embeddings in parallel
    # We load all into memory since we need to build the index.
    # If memory is an issue, we might need a different approach (e.g., adding to index in batches),
    # but for standard ImageNet features (1.2M * 1024 * 4 bytes ~= 5GB), it fits in RAM.

    # Check dimension from first file
    first_emb = load_npy_file(file_paths[0])
    if first_emb is None:
        logger.error("Could not load first file to determine dimensions.")
        return

    # Handle (1, D) vs (D,) shapes
    embedding_dim = first_emb.shape[-1]
    logger.info("Detected embedding dimension: %d", embedding_dim)

    # Use ThreadPoolExecutor for I/O bound task
    logger.info("Loading features with %d workers...", num_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # We can use map, but we want a progress bar
        # Converting to list to force execution for progress bar if we wrap it
        # But map returns an iterator.

        # Using list(executor.map(...)) is simplest but no progress bar easily without tqdm wrapping the iterator
        try:
            from tqdm import tqdm

            results = list(tqdm(executor.map(load_npy_file, file_paths), total=len(file_paths), unit="file"))
        except ImportError:
            results = list(executor.map(load_npy_file, file_paths))

    # Filter failures
    valid_embeddings = [r for r in results if r is not None]
    if len(valid_embeddings) < len(file_paths):
        logger.warning("Failed to load %d files.", len(file_paths) - len(valid_embeddings))

    # Stack
    logger.info("Stacking embeddings...")
    embeddings = np.vstack(valid_embeddings).astype(np.float32)

    # Verify shape
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, embedding_dim)

    logger.info("Final embeddings shape: %s", str(embeddings.shape))

    # 4. Normalize for Cosine Similarity
    logger.info("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # 5. Create FAISS Index
    logger.info("Creating FAISS index of type '%s'...", index_type)

    if index_type == "flat":
        # Flat index for exact search (inner product for normalized vectors)
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
    elif index_type == "ivf":
        # IVF index for approximate search
        # Ensure we have enough training data (at least 30 * nlist points)
        min_training_points = max(30 * nlist, 1000)
        if embeddings.shape[0] < min_training_points:
            logger.warning(
                "Only %d training points available, need at least %d for %d clusters",
                embeddings.shape[0],
                min_training_points,
                nlist,
            )
            # Reduce nlist to be more appropriate for the data size
            adjusted_nlist = max(1, min(nlist, embeddings.shape[0] // 30))
            if adjusted_nlist != nlist:
                logger.info("Adjusting nlist from %d to %d", nlist, adjusted_nlist)
                nlist = adjusted_nlist

        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        index.nprobe = nprobe  # Number of clusters to search

        # Train the index
        logger.info("Training IVF index with %d clusters...", nlist)
        # Use a subset for training if dataset is huge, but here we use all or a subset
        training_data = embeddings.astype(np.float32)
        index.train(training_data)

        # Add vectors
        logger.info("Adding vectors to IVF index...")
        index.add(embeddings)

    elif index_type == "hnsw":
        # HNSW index for approximate search
        index = faiss.IndexHNSWFlat(embedding_dim, 32)
        index.hnsw.efConstruction = 200  # Construction parameter
        index.hnsw.efSearch = 50  # Search parameter
        logger.info("Adding vectors to HNSW index...")
        index.add(embeddings)

    elif index_type == "ivfpq":
        # IVF PQ index for memory-efficient approximate search
        # Check if embedding dimension is divisible by pq_m
        if embedding_dim % pq_m != 0:
            raise ValueError(f"Embedding dimension {embedding_dim} must be divisible by pq_m {pq_m}")

        # Ensure we have enough training data (at least 30 * nlist points)
        min_training_points = max(30 * nlist, 1000)
        if embeddings.shape[0] < min_training_points:
            logger.warning(
                "Only %d training points available, need at least %d for %d clusters",
                embeddings.shape[0],
                min_training_points,
                nlist,
            )
            # Reduce nlist to be more appropriate for the data size
            adjusted_nlist = max(1, min(nlist, embeddings.shape[0] // 30))
            if adjusted_nlist != nlist:
                logger.info("Adjusting nlist from %d to %d", nlist, adjusted_nlist)
                nlist = adjusted_nlist

        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, pq_m, pq_bits)
        index.nprobe = nprobe  # Number of clusters to search

        # Train the index
        logger.info("Training IVF PQ index with %d clusters, %d sub-vectors, %d bits...", nlist, pq_m, pq_bits)
        training_data = embeddings.astype(np.float32)
        index.train(training_data)

        # Add vectors
        logger.info("Adding vectors to IVF PQ index...")
        index.add(embeddings)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    # 6. Save Index
    os.makedirs(output_dir, exist_ok=True)
    # Update output filename if default to include index type
    if index_name == "imagenet_index.faiss" and index_type != "flat":
        base, ext = os.path.splitext(index_name)
        index_name = f"{base}_{index_type}{ext}"

    output_path = os.path.join(output_dir, index_name)
    logger.info("Saving index to: %s", output_path)
    faiss.write_index(index, output_path)

    # Save Config
    config_path = os.path.join(output_dir, f"{os.path.splitext(index_name)[0]}_config.json")
    config = {
        "index_type": index_type,
        "embedding_dim": embedding_dim,
        "num_vectors": index.ntotal,
        "source_metadata": metadata_path,
        "feature_key": feature_key,
        "created": str(np.datetime64("now")),
        "parameters": {
            "nlist": nlist if index_type in ["ivf", "ivfpq"] else None,
            "pq_m": pq_m if index_type == "ivfpq" else None,
            "pq_bits": pq_bits if index_type == "ivfpq" else None,
            "nprobe": nprobe if index_type in ["ivf", "ivfpq"] else None,
        },
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config to: %s", config_path)

    elapsed = time.time() - start_time
    logger.info("Done! Index created with %d vectors in %.2f seconds.", index.ntotal, elapsed)


def main():
    parser = argparse.ArgumentParser(description="Create FAISS index from ImageNet metadata")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.json")
    parser.add_argument(
        "--feature_key",
        type=str,
        default="dinov2-l",
        help="Feature type to index (e.g., 'dinov2-l', 'clip-b'). Default: dinov2-l",
    )
    parser.add_argument("--output_dir", type=str, default="imagenet_index", help="Directory to save the FAISS index")
    parser.add_argument("--index_name", type=str, default="imagenet_index.faiss", help="Filename for the FAISS index")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker threads for loading files")

    parser.add_argument(
        "--index_type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw", "ivfpq"],
        help="Type of FAISS index to create (default: flat)",
    )
    parser.add_argument("--nlist", type=int, default=100, help="Number of clusters for IVF index (default: 100)")
    parser.add_argument(
        "--pq_m", type=int, default=8, help="Number of sub-vectors for Product Quantization (default: 8)"
    )
    parser.add_argument("--pq_bits", type=int, default=8, help="Number of bits per sub-vector for PQ (default: 8)")
    parser.add_argument(
        "--nprobe", type=int, default=10, help="Number of clusters to search during query (default: 10)"
    )

    args = parser.parse_args()

    create_faiss_index(
        metadata_path=args.metadata,
        feature_key=args.feature_key,
        output_dir=args.output_dir,
        index_name=args.index_name,
        num_workers=args.num_workers,
        index_type=args.index_type,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_bits=args.pq_bits,
        nprobe=args.nprobe,
    )


if __name__ == "__main__":
    main()
