# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import logging
import random
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler, DataLoader
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("GroupDiff")


class MetadataImageDataset(Dataset):
    """
    Dataset that loads images from disk based on metadata.json.

    This replaces ImageFolder and loads images based on metadata containing:
    - samples: list of dicts with 'id' (relative path), 'class', and optionally 'features'
    - class_to_samples: dict mapping class names to lists of sample IDs
    - class_to_idx: dict mapping class names to integer indices

    Args:
        metadata_path (str): Path to metadata.json file
        image_root (str): Root directory containing images (IDs in metadata are relative to this)
        transform (callable | None): Optional transform to apply to images
        load_latent (bool): Whether to load VAE latents
        latent_feature_name (str | None): Name of latent feature in metadata (e.g., "vae_latent")
        latent_root (str | None): Root directory for latent files (defaults to image_root)
        load_features (bool): Whether to load additional features
        feature_names (list[str] | None): List of feature names to load from metadata
        features_root (str | None): Root directory for feature files (defaults to image_root)
        load_image (bool): Whether to load raw images (default: True)

    Returns:
        dict[str, Any]: Dictionary with keys:
            - img: transformed image (None if load_image=False)
            - label: class label (int)
            - latent: VAE latent (None if not loaded)
            - feats: dict of features (None if not loaded)

    Raises:
        FileNotFoundError: If metadata_path or required files don't exist
        ValueError: If metadata format is invalid
    """

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        transform: Any | None = None,
        load_latent: bool = False,
        latent_feature_name: str | None = None,
        latent_root: str | None = None,
        load_features: bool = False,
        feature_names: list[str] | None = None,
        features_root: str | None = None,
        load_image: bool = True,
    ):
        """Initialize dataset from metadata file."""
        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Validate metadata structure
        if "samples" not in metadata:
            raise ValueError("Metadata must contain 'samples' field")
        if "class_to_idx" not in metadata:
            raise ValueError("Metadata must contain 'class_to_idx' field")

        self.metadata = metadata
        self.samples = metadata["samples"]
        self.class_to_idx = metadata["class_to_idx"]
        self.image_root = image_root
        self.transform = transform

        # Latent loading configuration
        self.load_image = load_image
        self.load_latent = load_latent
        self.latent_feature_name = latent_feature_name
        self.latent_root = latent_root if latent_root is not None else image_root

        # Features loading configuration
        self.load_features = load_features
        self.feature_names = feature_names if feature_names is not None else []
        self.features_root = features_root if features_root is not None else image_root

        # Build ID to index mapping for fast lookup
        self.id_to_idx = {sample["id"]: idx for idx, sample in enumerate(self.samples)}

        logger.info(
            "Loaded MetadataImageDataset: %d samples, %d classes, load_image=%s, load_latent=%s, load_features=%s",
            len(self.samples),
            len(self.class_to_idx),
            load_image,
            load_latent,
            load_features,
        )

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        """
        Load and return a sample.

        Args:
            idx (int | str): Integer index or string ID of sample to load

        Returns:
            dict[str, Any]: Dictionary containing:
                - img: transformed image (PIL.Image or torch.Tensor depending on transform, None if load_image=False)
                - label: class label (int)
                - latent: VAE latent (np.ndarray or torch.Tensor, None if not loaded)
                - feats: dict of feature name -> feature array (None if not loaded)

        Raises:
            IndexError: If idx is out of bounds
            KeyError: If string ID not found
            FileNotFoundError: If required files don't exist
        """
        # Handle both integer indices (for DataLoader) and string IDs (for cluster sampling)
        if isinstance(idx, str):
            if idx not in self.id_to_idx:
                raise KeyError(f"Sample ID not found: {idx}")
            idx = self.id_to_idx[idx]

        if not 0 <= idx < len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.samples)}")

        sample = self.samples[idx]
        image_id = sample["id"]
        class_label = int(sample["class"])

        # Initialize result dictionary
        result = {
            "img": None,
            "label": class_label,
            "latent": None,
            "feats": None,
        }

        # Load image if configured
        if self.load_image:
            image_path = os.path.join(self.image_root, image_id)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = Image.open(image_path).convert("RGB")

            # Apply transform
            if self.transform is not None:
                image = self.transform(image)

            result["img"] = image

        # Load latent if configured
        if self.load_latent and self.latent_feature_name:
            if "features" in sample and self.latent_feature_name in sample["features"]:
                latent_rel_path = sample["features"][self.latent_feature_name]
                latent_path = os.path.join(self.latent_root, latent_rel_path)
                if os.path.exists(latent_path):
                    try:
                        latent = np.load(latent_path)
                        # Convert to torch tensor if needed
                        if isinstance(latent, np.ndarray):
                            latent = torch.from_numpy(latent)
                        result["latent"] = latent
                    except Exception as e:
                        logger.warning("Failed to load latent from %s: %s", latent_path, e)
                else:
                    logger.debug("Latent file not found: %s", latent_path)

        # Load features if configured
        if self.load_features and self.feature_names:
            feats = {}
            for feat_name in self.feature_names:
                if "features" in sample and feat_name in sample["features"]:
                    feat_rel_path = sample["features"][feat_name]
                    feat_path = os.path.join(self.features_root, feat_rel_path)
                    if os.path.exists(feat_path):
                        try:
                            feat = np.load(feat_path)
                            feats[feat_name] = feat
                        except Exception as e:
                            logger.warning("Failed to load feature %s from %s: %s", feat_name, feat_path, e)
                    else:
                        logger.debug("Feature file not found: %s", feat_path)
            result["feats"] = feats if feats else None

        return result

    @property
    def class_to_samples(self) -> dict[str, list[str]]:
        """Get class to samples mapping from metadata."""
        return self.metadata.get("class_to_samples", {})


def create_epoch_plan(
    all_indices: list[Any],
    query_function: Any,  # callable function
    sequence_length: int,
    sequence_ratio: float = 0.1,
    total_samples_per_batch: int = 32,
    num_workers: int = 1,  # Number of threads for parallel processing
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Create a comprehensive, non-repeating batch plan for the entire epoch.
    Ensures direct batch and sequence batch contain the same number of total samples.

    Args:
        all_indices (list[Any]): All indices of the dataset (can be int or string IDs).
        query_function (callable): Query function taking an anchor_idx and returning a list of candidate indices.
        sequence_length (int): Total length of each sequence (N).
        sequence_ratio (float): Ratio of data used as sequence "anchors".
        total_samples_per_batch (int): Total number of samples per batch (same for direct and sequence batch).
        num_workers (int): Number of threads to use for parallel query processing.

    Returns:
        tuple[list[dict], list[dict]]: (sequence_batches, direct_batches)
            - sequence_batches: List of sequence batches, each is a dict containing type, ids, and sequences
            - direct_batches: List of direct load batches, each is a dict containing type and ids
    """
    logger.info("Creating epoch plan with %d total indices", len(all_indices))

    # Skip sequence creation when sequence_length is 1
    if sequence_length == 1:
        logger.info("Sequence length is 1, skipping sequence creation and returning all samples as direct batches")
        # Create direct batches from all indices
        shuffled_indices = list(all_indices)
        random.shuffle(shuffled_indices)

        direct_batches = []
        for i in range(0, len(shuffled_indices), total_samples_per_batch):
            batch_ids = shuffled_indices[i : i + total_samples_per_batch]
            # Only add full batches (drop_last=True behavior)
            if len(batch_ids) == total_samples_per_batch:
                direct_batches.append({"type": "direct", "ids": batch_ids})

        num_dropped = len(shuffled_indices) % total_samples_per_batch
        if num_dropped > 0:
            logger.info("Dropped %d samples from incomplete last batch", num_dropped)
        logger.info(
            "Created %d direct batches with batch size %d (no sequences)", len(direct_batches), total_samples_per_batch
        )
        return [], direct_batches

    # Calculate batch configuration
    sequences_per_batch = total_samples_per_batch // sequence_length
    if total_samples_per_batch % sequence_length != 0:
        logger.warning(
            "total_samples_per_batch (%d) is not divisible by sequence_length (%d). "
            "Adjusting to %d sequences per batch.",
            total_samples_per_batch,
            sequence_length,
            sequences_per_batch,
        )

    effective_samples_per_sequence_batch = sequences_per_batch * sequence_length
    logger.info(
        "Batch configuration: %d sequences per batch, %d samples per sequence batch, %d samples per direct batch",
        sequences_per_batch,
        effective_samples_per_sequence_batch,
        total_samples_per_batch,
    )

    # 1. Shuffle all indices
    shuffled_indices = list(all_indices)
    random.shuffle(shuffled_indices)

    # 2. Split into "anchor pool" and "available pool"
    num_anchors = int(len(shuffled_indices) * sequence_ratio / sequence_length * 1.5)
    num_needed_sequences = int(len(shuffled_indices) * sequence_ratio / sequence_length)
    anchor_pool = shuffled_indices[:num_anchors]
    # "available pool" contains all indices that can be "retrieved" or used for "direct loading"
    available_pool = set(shuffled_indices[num_anchors:])

    logger.info("Anchor pool size: %d, Available pool size: %d", len(anchor_pool), len(available_pool))

    # --- 3. Greedy algorithm to build sequences ---
    individual_sequences = []

    # Helper function to process a chunk of anchors
    def process_anchor_chunk(chunk_anchors):
        results = []
        for anchor in chunk_anchors:
            try:
                candidates = query_function(anchor)
                results.append((anchor, candidates))
            except Exception as e:
                logger.warning("Query function failed for anchor %s: %s", anchor, e)
                results.append((anchor, []))
        return results

    # Use ThreadPoolExecutor for parallel processing if num_workers > 1
    if num_workers > 1:
        logger.info("Using %d threads for query processing", num_workers)
        chunk_size = max(1, len(anchor_pool) // (num_workers * 4))
        chunks = [anchor_pool[i : i + chunk_size] for i in range(0, len(anchor_pool), chunk_size)]

        anchor_candidates_map = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_anchor_chunk, chunk) for chunk in chunks]
            for future in tqdm(as_completed(futures), desc="Querying candidates"):
                for anchor, candidates in future.result():
                    anchor_candidates_map[anchor] = candidates
    else:
        # Sequential processing
        anchor_candidates_map = {}
        for anchor in tqdm(anchor_pool, desc="Querying candidates"):
            try:
                anchor_candidates_map[anchor] = query_function(anchor)
            except Exception as e:
                logger.warning("Query function failed for anchor %s: %s", anchor, e)
                anchor_candidates_map[anchor] = []

    # Now build sequences sequentially to maintain deterministic behavior with availability checks
    # Although parallel querying is done, allocation needs to be sequential to respect 'available_pool'
    used_in_sequences = set()

    for anchor in tqdm(anchor_pool, desc="Building sequences"):
        retrieved_candidates = anchor_candidates_map.get(anchor, [])

        # b. Filter valid candidates in "available pool"
        valid_candidates = [idx for idx in retrieved_candidates if idx in available_pool]

        # c. If enough valid candidates, build sequence
        if len(valid_candidates) >= sequence_length - 1:
            retrieved_items = valid_candidates[: sequence_length - 1]

            # Build single sequence
            current_sequence = [anchor] + retrieved_items
            individual_sequences.append(current_sequence)

            # d. Update status to ensure no reuse
            used_in_sequences.add(anchor)
            for item in retrieved_items:
                used_in_sequences.add(item)

            # Remove retrieved items from available pool
            available_pool.difference_update(retrieved_items)
        else:
            # drop the sample and put it into the available pool
            available_pool.add(anchor)

    # Limit number of sequences
    if len(individual_sequences) > num_needed_sequences:
        # Put unused sequence samples back to available pool
        for no_use_sequence in individual_sequences[num_needed_sequences:]:
            for sample in no_use_sequence:
                available_pool.add(sample)

        individual_sequences = individual_sequences[:num_needed_sequences]

    logger.info("Created %d individual sequences", len(individual_sequences))

    # --- 4. Combine individual sequences into sequence batches ---
    sequence_batches = []
    for i in range(0, len(individual_sequences), sequences_per_batch):
        batch_sequences = individual_sequences[i : i + sequences_per_batch]
        if len(batch_sequences) == sequences_per_batch:  # Only keep full batches
            # Flatten sequences to get ids
            flat_ids = [item for seq in batch_sequences for item in seq]
            sequence_batch = {
                "type": "sequence",
                "ids": flat_ids,
                "sequences": batch_sequences,
                "sequences_per_batch": sequences_per_batch,
                "sequence_length": sequence_length,
            }
            sequence_batches.append(sequence_batch)
        else:
            # Put unused partial batch sequences back to available pool
            for seq in batch_sequences:
                for sample in seq:
                    available_pool.add(sample)

    logger.info("Created %d sequence batches with %d sequences each", len(sequence_batches), sequences_per_batch)

    # --- 5. Build direct batches with remaining indices ---
    direct_pool = list(available_pool)
    random.shuffle(direct_pool)

    # Split into batches of specified size
    direct_batches = []
    for i in range(0, len(direct_pool), total_samples_per_batch):
        batch_ids = direct_pool[i : i + total_samples_per_batch]
        # Only add full batches (drop_last=True behavior)
        if len(batch_ids) == total_samples_per_batch:
            direct_batches.append({"type": "direct", "ids": batch_ids})

    num_dropped = len(direct_pool) % total_samples_per_batch
    if num_dropped > 0:
        logger.info("Dropped %d samples from incomplete direct batch", num_dropped)
    logger.info("Created %d direct batches with batch size %d", len(direct_batches), total_samples_per_batch)
    logger.info(
        "Total samples used: %d, Total samples available: %d",
        len(used_in_sequences) + len(direct_pool),
        len(all_indices),
    )

    # Enhanced logging for debugging
    total_batches = len(sequence_batches) + len(direct_batches)
    sequence_ratio_actual = len(sequence_batches) / total_batches if total_batches > 0 else 0
    logger.info(
        "EPOCH PLAN SUMMARY: %d total batches (%d sequence, %d direct), sequence ratio: %.3f",
        total_batches,
        len(sequence_batches),
        len(direct_batches),
        sequence_ratio_actual,
    )

    return sequence_batches, direct_batches


def partition_batch_plan(batch_plan: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]]:
    """
    Split the total plan into parts for each GPU to support DDP.
    Ensures different GPUs get different batches.
    """
    # Ensure total batches is divisible by world_size to prevent deadlocks in training loop
    # caused by unequal batch counts leading to mismatched collective operations
    if len(batch_plan) % world_size != 0:
        num_batches_to_keep = (len(batch_plan) // world_size) * world_size
        logger.debug(
            "Dropping %d batches to ensure even DDP splitting (total: %d, world_size: %d)",
            len(batch_plan) - num_batches_to_keep,
            len(batch_plan),
            world_size,
        )
        batch_plan = batch_plan[:num_batches_to_keep]

    # Simple round-robin allocation (or modulo allocation)
    my_plan = [b for i, b in enumerate(batch_plan) if i % world_size == rank]
    return my_plan


class PredefinedBatchSampler(Sampler):
    """
    A Sampler that yields batches based on a predefined plan (list of dicts).
    Each dict in the plan must have an 'ids' key containing the list of sample indices for that batch.
    """

    def __init__(self, batch_plan: list[dict[str, Any]]):
        """
        batch_plan: Predefined list, e.g.:
        [{'ids': [1, 2, 3], 'type': 'single'}, {'ids': [4, 5], 'type': 'group'}]
        """
        self.batch_plan = batch_plan

    def __iter__(self) -> Iterator[list[Any]]:
        # Yield the 'ids' list directly
        # DataLoader receives this list and calls dataset[id] for each id
        for batch_info in self.batch_plan:
            yield batch_info["ids"]

    def __len__(self):
        return len(self.batch_plan)


# Backward compatibility / DDP Wrapper
class DDPPrecomputedPlanSampler(PredefinedBatchSampler):
    """
    DDP-aware wrapper for PredefinedBatchSampler.
    Takes a full global plan, partitions it for the current rank, and behaves as a Sampler.
    """

    def __init__(
        self,
        epoch_plan: list[dict[str, Any]],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,  # Shuffle the plan itself?
        seed: int = 0,
        epoch: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
            else:
                try:
                    num_replicas = dist.get_world_size()
                except Exception:
                    num_replicas = 1

        if rank is None:
            if not dist.is_available():
                rank = 0
            else:
                try:
                    rank = dist.get_rank()
                except Exception:
                    rank = 0

        self.global_plan = epoch_plan
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

        # Prepare local plan
        self.local_plan: list[dict[str, Any]] = []
        self.update_local_plan()

        # Initialize parent with local plan
        super().__init__(self.local_plan)

    def update_local_plan(self):
        # Shuffle global plan if needed (deterministically across ranks)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Shuffle indices of the plan
            indices = torch.randperm(len(self.global_plan), generator=g).tolist()
            shuffled_plan = [self.global_plan[i] for i in indices]
        else:
            shuffled_plan = self.global_plan

        # Partition for this rank
        self.local_plan = partition_batch_plan(shuffled_plan, self.rank, self.num_replicas)
        self.batch_plan = self.local_plan  # Update parent's plan reference

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.update_local_plan()


class MetaDataLoader:
    """
    Wrapper around DataLoader to inject metadata from the batch plan back into the batch.
    """

    def __init__(self, dataloader: DataLoader, batch_plan: list[dict[str, Any]]):
        self.dataloader = dataloader
        self.batch_plan = batch_plan
        self.iter_loader = None

    def __iter__(self):
        # Iterate both plan and loader
        # Because BatchSampler is deterministic, the order is guaranteed to match
        for plan_item, batch_data in zip(self.batch_plan, self.dataloader):
            # plan_item contains: {'ids': [...], 'type': '...', ...}
            # batch_data contains: {'images': tensor, 'ids': [...]} (from collate_fn)

            # Inject metadata
            if isinstance(batch_data, dict):
                batch_data["meta_type"] = plan_item.get("type")
                batch_data["meta_info"] = plan_item  # attach full info if needed

                # Handle Sequence Batch Reshaping
                if batch_data["meta_type"] == "sequence":
                    seq_len = plan_item.get("sequence_length", 1)
                    
                    # Handle pixel_values (online tokenization)
                    if "pixel_values" in batch_data:
                        imgs = batch_data["pixel_values"]
                        # Check if we need to reshape [B*S, C, H, W] -> [B, S, C, H, W]
                        # But only reshape if imgs are 4D (not already grouped)
                        if seq_len > 1 and imgs.ndim == 4 and imgs.shape[0] % seq_len == 0:
                            b = imgs.shape[0] // seq_len
                            # Reshape pixel_values
                            batch_data["pixel_values"] = imgs.view(b, seq_len, *imgs.shape[1:])

                            # Reshape labels if present
                            if "labels" in batch_data:
                                labels = batch_data["labels"]
                                if labels.shape[0] == imgs.shape[0]:
                                    batch_data["labels"] = labels.view(b, seq_len, *labels.shape[1:])
                    
                    # Handle latents (cached tokens)
                    if "latents" in batch_data:
                        latents = batch_data["latents"]
                        # Check if we need to reshape [B*S, C, H, W] -> [B, S, C, H, W]
                        # But only reshape if latents are 4D (not already grouped)
                        if seq_len > 1 and latents.ndim == 4 and latents.shape[0] % seq_len == 0:
                            b = latents.shape[0] // seq_len
                            # Reshape latents
                            batch_data["latents"] = latents.view(b, seq_len, *latents.shape[1:])

                            # Reshape labels if present and not already reshaped
                            if "labels" in batch_data:
                                labels = batch_data["labels"]
                                if labels.ndim == 1 and labels.shape[0] == latents.shape[0]:
                                    batch_data["labels"] = labels.view(b, seq_len)
            yield batch_data

    def __len__(self):
        return len(self.dataloader)


# --- Metadata Loading and Query Helpers ---


def load_metadata_from_json(json_path: str) -> dict[str, Any]:
    """Load metadata JSON file created by extract_feats.py"""
    with open(json_path) as f:
        return json.load(f)


class ClassQueryFunction:
    """
    Query function based on class ID.
    Given an anchor, returns other samples with the same class.
    """

    def __init__(self, metadata: dict[str, Any]):
        self.class_to_samples = metadata.get("class_to_samples", {})
        # Build reverse mapping: id -> class
        self.id_to_class = {}
        # metadata['class_to_samples'] maps class_name (str) -> list of ids
        for cls_name, samples in self.class_to_samples.items():
            for s in samples:
                self.id_to_class[s] = cls_name

    def __call__(self, anchor_id: str) -> list[str]:
        cls = self.id_to_class.get(anchor_id)
        if cls is None:
            return []
        # Return all samples of this class
        # Note: create_epoch_plan handles removing used items, so returning all is fine.
        return self.class_to_samples.get(cls, [])


class FaissQueryFunction:
    """
    Query function based on FAISS similarity search.
    Requires a pre-built FAISS index and access to feature files (.npy).
    """

    def __init__(self, index_path: str, metadata: dict[str, Any], feature_name: str, features_root: str | None = None):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for FaissQueryFunction. Install with `pip install faiss-cpu` or `faiss-gpu`."
            )

        logger.info("Loading FAISS index from %s", index_path)
        self.index = faiss.read_index(index_path)

        # Load FAISS config if available and set parameters
        base_name = os.path.splitext(index_path)[0]
        config_path = f"{base_name}_config.json"
        if os.path.exists(config_path):
            logger.info("Loading FAISS config from %s", config_path)
            with open(config_path) as f:
                self.config = json.load(f)

            # Set index parameters from config if available
            if "parameters" in self.config:
                params = self.config["parameters"]
                if "nprobe" in params and params["nprobe"] is not None:
                    self.index.nprobe = params["nprobe"]
                    logger.info("Set index nprobe to %d", params["nprobe"])

                # For IVFPQ or other types, we might want to log other params
                if "nlist" in params:
                    logger.info("Index uses nlist=%s", params["nlist"])
                if "pq_m" in params:
                    logger.info("Index uses pq_m=%s", params["pq_m"])
                if "pq_bits" in params:
                    logger.info("Index uses pq_bits=%s", params["pq_bits"])
        else:
            logger.warning("FAISS config file not found at %s", config_path)
            self.config = {}

        self.metadata = metadata
        self.feature_name = feature_name
        self.features_root = features_root

        # Map ID -> feature relative path
        self.id_to_feature_path = {}
        for s in metadata.get("samples", []):
            # s is {'id': ..., 'features': {'feat_name': 'path', ...}}
            if "features" in s and feature_name in s["features"]:
                self.id_to_feature_path[s["id"]] = s["features"][feature_name]

        # Map Index (in FAISS) -> ID
        # Assumption: FAISS index was built using samples in the same order as metadata['samples']
        # IMPORTANT: This assumption must hold true, otherwise results will be wrong
        # The create_fasiss.py script iterates metadata['samples'] in order, so it should be correct

        samples = metadata.get("samples", [])

        # We only care about samples that actually have this feature
        # This matches the logic in create_fasiss.py where we filter by feature existence
        feature_samples = [s for s in samples if "features" in s and feature_name in s["features"]]

        self.idx_to_id = {i: s["id"] for i, s in enumerate(feature_samples)}
        self.num_vectors = len(feature_samples)

        logger.info(
            "Initialized FaissQueryFunction with %d vectors (index size: %d)", self.num_vectors, self.index.ntotal
        )

        if self.num_vectors != self.index.ntotal:
            logger.warning(
                "Number of samples with feature '%s' (%d) does not match FAISS index size (%d). "
                "This might indicate a mismatch between metadata and index.",
                feature_name,
                self.num_vectors,
                self.index.ntotal,
            )

    def __call__(self, anchor_id: str) -> list[str]:
        feat_rel_path = self.id_to_feature_path.get(anchor_id)
        if not feat_rel_path:
            return []

        # Resolve full path
        if self.features_root:
            full_path = os.path.join(self.features_root, feat_rel_path)
        else:
            # If root not provided, assume relative to CWD or absolute if path is absolute
            full_path = feat_rel_path

        try:
            # Load vector using logic from create_fasiss.py
            # Check if file exists first
            if not os.path.exists(full_path):
                return []

            vec = np.load(full_path)

            # Faiss expects (n, d) float32
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            vec = vec.astype("float32")

            # Normalize if needed (FlatIP uses inner product, usually on normalized vectors)
            # Assuming create_fasiss.py normalized the vectors before indexing
            # We need to use faiss.normalize_L2, but faiss is imported inside __init__
            # So we need to import it here or use self.index (which is a faiss object) to get the module?
            # Better to import at module level or inside method.
            import faiss

            faiss.normalize_L2(vec)

            # Search
            k = 100  # Retrieve enough candidates
            dist_matrix, indices_out = self.index.search(vec, k)

            # Map back to IDs
            candidates = []
            for idx in indices_out[0]:
                if idx != -1 and idx in self.idx_to_id:
                    candidates.append(self.idx_to_id[idx])

            return candidates

        except Exception as e:
            logger.warning("Error querying FAISS for anchor %s: %s", anchor_id, e)
            return []


class ClusterDataManager:
    """
    Manager class to handle cluster-based data loading configuration and plan generation.
    """

    def __init__(
        self,
        metadata_path: str,
        strategy: str = "class",  # "class" or "faiss"
        faiss_index_path: str | None = None,
        feature_name: str | None = None,
        features_root: str | None = None,
        sequence_length: int = 4,
        sequence_ratio: float = 0.1,
        total_samples_per_batch: int = 32,
        num_workers: int = 1,
        accelerator: Any | None = None,
    ):
        self.metadata = load_metadata_from_json(metadata_path)
        self.all_indices = [s["id"] for s in self.metadata["samples"]]

        if strategy == "class":
            self.query_fn = ClassQueryFunction(self.metadata)
        elif strategy == "faiss":
            if not faiss_index_path or not feature_name:
                raise ValueError("faiss_index_path and feature_name are required for faiss strategy")
            self.query_fn = FaissQueryFunction(faiss_index_path, self.metadata, feature_name, features_root)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.sequence_length = sequence_length
        self.sequence_ratio = sequence_ratio
        self.total_samples_per_batch = total_samples_per_batch
        self.num_workers = num_workers
        self.accelerator = accelerator

    def get_epoch_plan(self) -> list[dict[str, Any]]:
        """
        Generate the full batch plan for one epoch.
        Ensures consistency across processes by generating on main process and broadcasting.
        """
        # Determine if we are the main process
        is_main_process = True
        if self.accelerator is not None:
            is_main_process = self.accelerator.is_main_process

        epoch_plan = None
        if is_main_process:
            sequence_batches, direct_batches = create_epoch_plan(
                self.all_indices,
                self.query_fn,
                self.sequence_length,
                self.sequence_ratio,
                self.total_samples_per_batch,
                num_workers=self.num_workers,
            )
            epoch_plan = sequence_batches + direct_batches

        # Broadcast plan if distributed
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            # Use broadcast_object_list to send the plan from rank 0 to others
            object_list = [epoch_plan]
            dist.broadcast_object_list(object_list, src=0)
            epoch_plan = object_list[0]

        if epoch_plan is None:
            # Should only happen if something went wrong with broadcast or logic
            logger.warning("Epoch plan is None after generation/broadcast, falling back to local generation")
            sequence_batches, direct_batches = create_epoch_plan(
                self.all_indices,
                self.query_fn,
                self.sequence_length,
                self.sequence_ratio,
                self.total_samples_per_batch,
                num_workers=self.num_workers,
            )
            epoch_plan = sequence_batches + direct_batches

        return epoch_plan

    def get_sampler(
        self,
        epoch_plan: list[dict[str, Any]],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        epoch: int = 0,
    ) -> DDPPrecomputedPlanSampler:
        return DDPPrecomputedPlanSampler(
            epoch_plan,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
        )

    def wrap_dataloader(self, dataloader: DataLoader, batch_plan: list[dict[str, Any]]) -> MetaDataLoader:
        return MetaDataLoader(dataloader, batch_plan)


class ImageNetDataset:
    """
    Main dataset class for ImageNet training that can be instantiated from config.

    This class handles:
    - Loading images from metadata
    - Optionally loading VAE latents
    - Optionally loading extracted features
    - Epoch planning with ClusterDataManager
    - Creating appropriate DataLoaders for training

    Args:
        root_dir (str): Root directory containing images
        metadata_path (str | None): Path to metadata.json file (auto-detected if None)
        image_size (int): Size to resize/crop images to
        seq_length (int): Sequence length for epoch planning
        global_batch_size (int): Global batch size across all processes
        use_epoch_planning (bool): Whether to use epoch planning
        group_ratio (float | None): Ratio of group/sequence batches to total batches (default: 0.1)
        sequence_ratio (float | None): Deprecated alias for group_ratio (for backward compatibility)
        num_workers (int): Number of DataLoader workers
        cluster_manager (dict | None): Config for ClusterDataManager
        transform (Any | None): Transform to apply to images
        load_latent (bool): Whether to load VAE latents
        latent_feature_name (str | None): Name of latent feature in metadata
        latent_root (str | None): Root directory for latent files
        load_features (bool): Whether to load additional features
        feature_names (list[str] | None): List of feature names to load
        features_root (str | None): Root directory for feature files
        load_image (bool): Whether to load raw images (default: True)

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If configuration is invalid
    """

    def __init__(
        self,
        root_dir: str,
        metadata_path: str | None = None,
        image_size: int = 256,
        seq_length: int = 4,
        global_batch_size: int = 256,
        use_epoch_planning: bool = True,
        group_ratio: float | None = None,
        sequence_ratio: float | None = None,
        num_workers: int = 8,
        cluster_manager: dict[str, Any] | None = None,
        transform: Any | None = None,
        load_latent: bool = False,
        latent_feature_name: str | None = None,
        latent_root: str | None = None,
        load_features: bool = False,
        feature_names: list[str] | None = None,
        features_root: str | None = None,
        load_image: bool = True,
        **kwargs: Any,
    ):
        """Initialize ImageNet dataset."""
        self.root_dir = root_dir
        self.image_size = image_size
        self.seq_length = seq_length
        self.global_batch_size = global_batch_size
        self.use_epoch_planning = use_epoch_planning

        # Handle group_ratio vs sequence_ratio (group_ratio is the new name, sequence_ratio for backward compat)
        if group_ratio is not None:
            self.sequence_ratio = group_ratio
        elif sequence_ratio is not None:
            self.sequence_ratio = sequence_ratio
        else:
            self.sequence_ratio = 0.1  # default value

        self.num_workers = num_workers
        self.cluster_manager_config = cluster_manager
        self.transform = transform

        # Latent and features configuration
        self.load_image = load_image
        self.load_latent = load_latent
        self.latent_feature_name = latent_feature_name if latent_feature_name else "vae_latent"
        # latent_root defaults to features_root, then to root_dir
        self.latent_root = latent_root if latent_root is not None else (features_root if features_root is not None else root_dir)
        self.load_features = load_features
        self.feature_names = feature_names
        self.features_root = features_root

        # Auto-detect metadata path if not provided
        if metadata_path is None:
            metadata_path = os.path.join(root_dir, "metadata.json")

        self.metadata_path = metadata_path

        # Validate metadata file exists
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}. Please run extract_feats.py to generate metadata.json"
            )

        # Default batch size per process (will be updated if accelerator is set)
        self.batch_size_per_process = self.global_batch_size

        # Create the underlying dataset
        logger.info("Creating MetadataImageDataset with metadata_path: %s", self.metadata_path)
        logger.info("Dataset loading configuration: load_image=%s, load_latent=%s, latent_feature_name=%s, latent_root=%s",
                   self.load_image, self.load_latent, self.latent_feature_name, self.latent_root)
        self.dataset = MetadataImageDataset(
            metadata_path=self.metadata_path,
            image_root=self.root_dir,
            transform=self.transform,
            load_latent=self.load_latent,
            latent_feature_name=self.latent_feature_name,
            latent_root=self.latent_root,
            load_features=self.load_features,
            feature_names=self.feature_names,
            features_root=self.features_root,
            load_image=self.load_image,
        )

        # Initialize cluster manager if epoch planning is enabled
        self.cluster_manager: ClusterDataManager | None = None
        if self.use_epoch_planning:
            self._init_cluster_manager()

    def set_accelerator(self, accelerator: Any) -> None:
        """
        Set accelerator to handle DDP batch sizing.

        This allows the dataset to be instantiated without DDP context initially,
        and then updated with the correct world size for batch planning.
        """
        self.accelerator = accelerator
        world_size = accelerator.num_processes
        if world_size > 1:
            self.batch_size_per_process = self.global_batch_size // world_size
            logger.info(
                "Accelerator set (world_size=%d). Adjusting batch size for ClusterDataManager: global=%d -> per_process=%d",
                world_size,
                self.global_batch_size,
                self.batch_size_per_process,
            )

            # Re-initialize cluster manager with correct batch size
            if self.use_epoch_planning:
                self._init_cluster_manager()

    def _init_cluster_manager(self) -> None:
        """Initialize ClusterDataManager from config."""
        # Get accelerator if available
        accelerator = getattr(self, "accelerator", None)

        if self.cluster_manager_config is None:
            # Use default class-based strategy
            logger.info("No cluster_manager config provided, using default class-based strategy")
            self.cluster_manager = ClusterDataManager(
                metadata_path=self.metadata_path,
                strategy="class",
                sequence_length=self.seq_length,
                sequence_ratio=self.sequence_ratio,
                total_samples_per_batch=self.batch_size_per_process,
                num_workers=self.num_workers,
                accelerator=accelerator,
            )
        else:
            # Instantiate from config dict
            logger.info("Instantiating ClusterDataManager from config")

            # Extract params from config and ensure it's a mutable dict
            if isinstance(self.cluster_manager_config, dict):
                # Get params or use the whole config, ensure it's a copy
                config_params = self.cluster_manager_config.get("params", self.cluster_manager_config)
                params = dict(config_params)  # Create a new mutable dict
            else:
                # Handle OmegaConf or object with attributes
                params = {}
                if hasattr(self.cluster_manager_config, "params"):
                    cfg_params = self.cluster_manager_config.params
                    # Convert to dict
                    if hasattr(cfg_params, "items"):
                        params = dict(cfg_params.items())
                    elif hasattr(cfg_params, "__dict__"):
                        params = dict(vars(cfg_params))
                elif hasattr(self.cluster_manager_config, "items"):
                    params = dict(self.cluster_manager_config.items())

            # Inject defaults if not present (using dict operations)
            if "metadata_path" not in params:
                params["metadata_path"] = self.metadata_path
            if "sequence_length" not in params:
                params["sequence_length"] = self.seq_length
            if "sequence_ratio" not in params:
                params["sequence_ratio"] = self.sequence_ratio
            if "total_samples_per_batch" not in params:
                params["total_samples_per_batch"] = self.batch_size_per_process
            if "num_workers" not in params:
                params["num_workers"] = self.num_workers
            if "accelerator" not in params:
                params["accelerator"] = accelerator

            # Create ClusterDataManager with params
            self.cluster_manager = ClusterDataManager(**params)

        logger.info("ClusterDataManager initialized successfully")

    def set_transform(self, transform: Any) -> None:
        """Set transform for the dataset."""
        self.transform = transform
        if hasattr(self, "dataset"):
            self.dataset.transform = transform

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        """Get item from dataset."""
        return self.dataset[idx]

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Get class to index mapping."""
        return self.dataset.class_to_idx

    @property
    def samples(self) -> list[dict[str, Any]]:
        """Get samples list."""
        return self.dataset.samples
