"""Multi-view spatial fusion of semantic object observations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from mapping.object_map import ObjectMap, SemanticObject, Vec3


def load_fusion_config(classes_config: Path) -> dict:
    with classes_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("fusion", {"cluster_distance_m": 0.5, "min_observations": 1})


def fuse_observations(
    observations: list[SemanticObject],
    cluster_distance_m: float = 0.5,
    min_observations: int = 1,
    agent_id: str = "robot_0",
) -> ObjectMap:
    """Cluster observations of the same class in 3D and merge them."""
    if not observations:
        return ObjectMap(agent_id=agent_id)

    by_class: dict[str, list[SemanticObject]] = {}
    for obs in observations:
        by_class.setdefault(obs.class_name, []).append(obs)

    fused_objects: list[SemanticObject] = []

    for class_name, class_obs in by_class.items():
        clusters = _cluster_spatial(class_obs, cluster_distance_m)
        for cluster in clusters:
            if len(cluster) < min_observations:
                continue
            merged = _merge_cluster(cluster)
            fused_objects.append(merged)

    return ObjectMap(objects=fused_objects, agent_id=agent_id)


def _cluster_spatial(
    observations: list[SemanticObject],
    distance_m: float,
) -> list[list[SemanticObject]]:
    """Simple greedy spatial clustering."""
    if not observations:
        return []

    clusters: list[list[SemanticObject]] = []
    used = [False] * len(observations)

    for i, obs in enumerate(observations):
        if used[i]:
            continue
        cluster = [obs]
        used[i] = True
        pi = _pos_array(obs)

        for j in range(i + 1, len(observations)):
            if used[j]:
                continue
            pj = _pos_array(observations[j])
            if np.linalg.norm(pi - pj) <= distance_m:
                cluster.append(observations[j])
                used[j] = True

        clusters.append(cluster)

    return clusters


def _pos_array(obj: SemanticObject) -> np.ndarray:
    return np.array([obj.position.x, obj.position.y, obj.position.z])


def _merge_cluster(cluster: list[SemanticObject]) -> SemanticObject:
    """Merge cluster into single object with averaged position and boosted confidence."""
    weights = np.array([o.confidence for o in cluster])
    weights = weights / weights.sum()

    positions = np.array([[_pos_array(o)[k] for o in cluster] for k in range(3)])
    avg_pos = (positions * weights).sum(axis=1)

    avg_confidence = min(1.0, float(np.mean([o.confidence for o in cluster])) + 0.05 * len(cluster))
    latest = max(cluster, key=lambda o: o.last_seen_ts)

    merged = SemanticObject(
        id=cluster[0].id,
        class_name=cluster[0].class_name,
        position=Vec3(float(avg_pos[0]), float(avg_pos[1]), float(avg_pos[2])),
        confidence=avg_confidence,
        observations=sum(o.observations for o in cluster),
        last_seen_ts=latest.last_seen_ts,
        agent_id=latest.agent_id,
        bbox3d=latest.bbox3d,
    )
    return merged


def fuse_from_config(
    observations: list[SemanticObject],
    classes_config: Path,
    agent_id: str = "robot_0",
) -> ObjectMap:
    fusion_cfg = load_fusion_config(classes_config)
    return fuse_observations(
        observations,
        cluster_distance_m=fusion_cfg.get("cluster_distance_m", 0.5),
        min_observations=fusion_cfg.get("min_observations", 1),
        agent_id=agent_id,
    )
