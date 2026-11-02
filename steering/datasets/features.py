"""Inventory-based dynamic sites; native IDs are never positional public-site IDs."""

from __future__ import annotations

import numpy as np

from core.io import identifier

FEATURE_SCHEMA = "branch-close-float32-v1"


def inventory_for(event, model):
    explicit = event.get("site_inventory")
    if explicit is not None:
        sites = [dict(site) for site in explicit]
    else:
        shapes, dtypes = event.get("storage_shapes"), event.get("storage_dtypes")
        if not isinstance(shapes, dict) or not shapes or not isinstance(dtypes, dict):
            raise ValueError("Hidden event has no shape/dtype site inventory")
        layers = model.get("num_hidden_layers")
        if type(layers) is not int or layers < 1:
            raise ValueError("Model inventory must declare num_hidden_layers")
        sites = []
        for key in shapes:
            if not key.startswith("layer_") or not key[6:].isdigit():
                raise ValueError("Non-native site requires explicit semantic inventory")
            native_id = int(key[6:])
            if not 0 <= native_id <= layers:
                raise ValueError(f"Native site outside frozen model inventory: {key}")
            shape = shapes[key]
            if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1:
                raise ValueError(f"Hidden site must be a single vector: {key}")
            if model.get("hidden_size") is not None and shape[0] != model["hidden_size"]:
                raise ValueError("Hidden width differs from the frozen model inventory")
            sites.append(
                {
                    "id": str(native_id),
                    "key": key,
                    "width": shape[0],
                    "storage_dtype": str(np.dtype(dtypes[key])),
                    "semantic": "final_norm" if native_id == layers else "decoder_output",
                    "decoder_layer": None if native_id == layers else native_id,
                }
            )
    ids = []
    for site in sites:
        ids.append(identifier(str(site["id"])))
        if site.get("semantic") not in {"decoder_output", "final_norm"} or site.get("width", 0) < 1:
            raise ValueError("Invalid hidden site semantic inventory")
        if np.dtype(site["storage_dtype"]).kind != "f":
            raise ValueError("Hidden site storage must use floating point dtype")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate site ID in hidden inventory")
    return sorted(
        sites,
        key=lambda site: (not site["id"].isdigit(), int(site["id"]) if site["id"].isdigit() else site["id"]),
    )


def select_sites(inventory, requested=None):
    if requested is None or len(requested) == 0:
        return list(inventory)
    ids = [str(value) for value in requested]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate site selection")
    mapping = {site["id"]: site for site in inventory}
    if not set(ids) <= mapping.keys():
        raise ValueError(f"Unknown explicit site IDs: {sorted(set(ids) - mapping.keys())}")
    return [mapping[value] for value in ids]


def vectors(arrays, inventory):
    result = {}
    for site in inventory:
        key, width = site["key"], site["width"]
        if key not in arrays:
            raise ValueError(f"Tensor is missing frozen site: {key}")
        value = np.asarray(arrays[key])
        if value.shape != (width,) or value.dtype != np.dtype(site["storage_dtype"]):
            raise ValueError(f"Hidden site shape/dtype differs from frozen inventory: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"Hidden site has non-finite values: {key}")
        result[site["id"]] = value.astype(np.float32, copy=False)
    return result
