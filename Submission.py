import json
import re
import shutil
from pathlib import Path

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SOURCE_ADAPTER_PATH = Path(
    "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"
)

WORK_DIR = Path("/kaggle/working")
OUTPUT_ROOT = WORK_DIR / "adapter_candidates"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FORCED_FUSED_RANK = 32
BUILD_ALL_CANDIDATES = False

CANDIDATE_SPECS = [
    {
        "name": "dense_svd_model",
        "strategy": "dense_svd",
        "min_rank_per_component": 0,
        "component_priority": {},
        "drop_lm_head": False,
        "namespace_mode": "model",
    },
    {
        "name": "block_topk_floor4_model",
        "strategy": "block_topk",
        "min_rank_per_component": 4,
        "component_priority": {},
        "drop_lm_head": False,
        "namespace_mode": "model",
    },
    {
        "name": "block_topk_floor4_x_bias_model",
        "strategy": "block_topk",
        "min_rank_per_component": 4,
        "component_priority": {"x_proj": 1.15},
        "drop_lm_head": False,
        "namespace_mode": "model",
    },
    {
        "name": "block_topk_floor4_model_no_lm_head",
        "strategy": "block_topk",
        "min_rank_per_component": 4,
        "component_priority": {},
        "drop_lm_head": True,
        "namespace_mode": "model",
    },
    {
        "name": "block_topk_floor4_backbone",
        "strategy": "block_topk",
        "min_rank_per_component": 4,
        "component_priority": {},
        "drop_lm_head": False,
        "namespace_mode": "backbone",
    },
]

ACTIVE_CANDIDATE_NAME = "block_topk_floor4_model"
CANDIDATE_BY_NAME = {spec["name"]: spec for spec in CANDIDATE_SPECS}
ACTIVE_CANDIDATE = CANDIDATE_BY_NAME[ACTIVE_CANDIDATE_NAME]

with open(SOURCE_ADAPTER_PATH / "adapter_config.json") as f:
    source_adapter_config = json.load(f)

source_adapter_tensors = {}
with safe_open(
    SOURCE_ADAPTER_PATH / "adapter_model.safetensors",
    framework="pt",
    device="cpu",
) as f:
    for key in f.keys():
        source_adapter_tensors[key] = f.get_tensor(key)

source_base_names = sorted(
    {
        re.sub(r"\.lora_[AB]\.weight$", "", key)
        for key in source_adapter_tensors
    }
)

mamba_merge_layers = {}
for base_name in source_base_names:
    for proj_name in ("gate_proj", "x_proj"):
        if f".{proj_name}" in base_name:
            layer_prefix = base_name.rsplit(f".{proj_name}", 1)[0]
            mamba_merge_layers.setdefault(layer_prefix, {})[proj_name] = base_name

mamba_merge_bases = {
    item
    for layer in mamba_merge_layers.values()
    for item in layer.values()
}

print("Loaded tensors:", len(source_adapter_tensors))
print("Mamba fused layers:", len(mamba_merge_layers))
print("Source rank:", source_adapter_config["r"])

def rename_base_name(base_name: str, namespace_mode: str) -> str:
    if namespace_mode == "model":
        return base_name
    if namespace_mode == "backbone":
        return base_name.replace(
            "base_model.model.model",
            "base_model.model.backbone",
        )
    raise ValueError(f"Unknown namespace_mode: {namespace_mode}")


def rewrite_expert_base_name(base_name: str, expert_idx: int) -> str:
    if ".experts.w1" in base_name:
        return re.sub(
            r"\.experts\.w1",
            f".experts.{expert_idx}.up_proj",
            base_name,
        )
    if ".experts.w2" in base_name:
        return re.sub(
            r"\.experts\.w2",
            f".experts.{expert_idx}.down_proj",
            base_name,
        )
    raise ValueError(f"Unsupported expert base name: {base_name}")


def compute_target_modules(tensors: dict[str, torch.Tensor]) -> list[str]:
    target_modules = sorted(
        {
            key.removesuffix(".lora_A.weight").removesuffix(".lora_B.weight").rsplit(".", 1)[-1]
            for key in tensors
        }
    )
    return target_modules


def build_adapter_config(
    output_tensors: dict[str, torch.Tensor],
    candidate_spec: dict,
) -> dict:
    config = dict(source_adapter_config)
    config["base_model_name_or_path"] = BASE_MODEL_NAME
    config["inference_mode"] = True
    config["target_modules"] = compute_target_modules(output_tensors)
    config["r"] = FORCED_FUSED_RANK
    config["rank_pattern"] = {}
    config["alpha_pattern"] = {}
    config["task_type"] = config.get("task_type", "CAUSAL_LM")
    config["modules_to_save"] = None
    config["fan_in_fan_out"] = False
    config["bias"] = "none"
    config["init_lora_weights"] = True
    config["peft_type"] = "LORA"
    config["use_dora"] = config.get("use_dora", False)
    config["use_rslora"] = config.get("use_rslora", False)
    config["lora_dropout"] = float(config.get("lora_dropout", 0.0))

    if candidate_spec["drop_lm_head"]:
        config["target_modules"] = [
            module for module in config["target_modules"] if module != "lm_head"
        ]

    return config

def svd_from_factors(
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_b, r_b = torch.linalg.qr(lora_B.float())
    q_a, r_a = torch.linalg.qr(lora_A.float().T)
    core = r_b @ r_a.T
    u_core, s, vh_core = torch.linalg.svd(core, full_matrices=False)
    u = q_b @ u_core
    vh = vh_core @ q_a.T
    return u, s, vh


def compress_dense_svd(
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    u, s, vh = svd_from_factors(lora_B, lora_A)
    k = min(rank, s.shape[0])
    sqrt_s = torch.sqrt(s[:k])
    new_B = (u[:, :k] * sqrt_s.unsqueeze(0)).to(lora_B.dtype).contiguous()
    new_A = (sqrt_s.unsqueeze(1) * vh[:k, :]).to(lora_A.dtype).contiguous()
    retained_energy = float(s[:k].square().sum() / s.square().sum()) if s.numel() else 1.0
    retained_mass = float(s[:k].sum() / s.sum()) if s.numel() else 1.0
    return new_B, new_A, {
        "rank": int(k),
        "retained_energy": retained_energy,
        "retained_singular_mass": retained_mass,
    }


def allocate_block_ranks(
    singular_values_by_component: list[tuple[str, torch.Tensor]],
    total_rank: int,
    min_rank_per_component: int,
    component_priority: dict[str, float],
) -> dict[str, int]:
    ranks = {}
    cursors = {}

    for name, singular_values in singular_values_by_component:
        guaranteed = min(min_rank_per_component, singular_values.numel())
        ranks[name] = guaranteed
        cursors[name] = guaranteed

    remaining = total_rank - sum(ranks.values())
    if remaining < 0:
        raise ValueError(
            f"Rank budget {total_rank} is smaller than guaranteed floor "
            f"{min_rank_per_component} x {len(singular_values_by_component)}"
        )

    while remaining > 0:
        best_name = None
        best_value = None
        for name, singular_values in singular_values_by_component:
            cursor = cursors[name]
            if cursor >= singular_values.numel():
                continue
            priority = component_priority.get(name, 1.0)
            value = float(singular_values[cursor]) * priority
            if best_value is None or value > best_value:
                best_name = name
                best_value = value

        if best_name is None:
            break

        ranks[best_name] += 1
        cursors[best_name] += 1
        remaining -= 1

    return ranks


def compress_block_topk(
    components: list[tuple[str, torch.Tensor, torch.Tensor]],
    total_rank: int,
    min_rank_per_component: int,
    component_priority: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    component_svds = []
    row_offset = 0

    for comp_name, lora_A, lora_B in components:
        u, s, vh = svd_from_factors(lora_B, lora_A)
        component_svds.append(
            {
                "name": comp_name,
                "u": u,
                "s": s,
                "vh": vh,
                "lora_A_dtype": lora_A.dtype,
                "lora_B_dtype": lora_B.dtype,
                "row_start": row_offset,
                "row_end": row_offset + lora_B.shape[0],
            }
        )
        row_offset += lora_B.shape[0]

    rank_allocation = allocate_block_ranks(
        [(item["name"], item["s"]) for item in component_svds],
        total_rank=total_rank,
        min_rank_per_component=min_rank_per_component,
        component_priority=component_priority,
    )

    fused_out_dim = row_offset
    merged_B = torch.zeros(fused_out_dim, total_rank, dtype=torch.float32)
    merged_A_parts = []
    rank_offset = 0
    component_stats = []
    total_energy = 0.0
    kept_energy = 0.0

    for item in component_svds:
        rank = rank_allocation[item["name"]]
        singular_values = item["s"]
        comp_total_energy = float(singular_values.square().sum()) if singular_values.numel() else 0.0
        comp_kept_energy = float(singular_values[:rank].square().sum()) if rank else 0.0
        total_energy += comp_total_energy
        kept_energy += comp_kept_energy

        if rank > 0:
            sqrt_s = torch.sqrt(singular_values[:rank])
            comp_B = item["u"][:, :rank] * sqrt_s.unsqueeze(0)
            comp_A = sqrt_s.unsqueeze(1) * item["vh"][:rank, :]
            merged_B[
                item["row_start"] : item["row_end"],
                rank_offset : rank_offset + rank,
            ] = comp_B
            merged_A_parts.append(comp_A)
            rank_offset += rank

        component_stats.append(
            {
                "component": item["name"],
                "allocated_rank": int(rank),
                "available_rank": int(singular_values.numel()),
                "retained_energy": (
                    comp_kept_energy / comp_total_energy if comp_total_energy else 1.0
                ),
            }
        )

    if merged_A_parts:
        merged_A = torch.cat(merged_A_parts, dim=0).to(
            components[0][1].dtype
        ).contiguous()
    else:
        merged_A = torch.zeros(
            0,
            components[0][1].shape[1],
            dtype=components[0][1].dtype,
        )

    merged_B = merged_B[:, :rank_offset].to(components[0][2].dtype).contiguous()
    return merged_B, merged_A, {
        "rank": int(rank_offset),
        "retained_energy": kept_energy / total_energy if total_energy else 1.0,
        "min_rank_per_component": int(min_rank_per_component),
        "component_priority": component_priority,
        "component_stats": component_stats,
    }

def build_output_tensors(candidate_spec: dict) -> tuple[dict[str, torch.Tensor], list[dict]]:
    output_tensors = {}
    fused_stats = []

    for base_name in source_base_names:
        lora_A = source_adapter_tensors[f"{base_name}.lora_A.weight"]
        lora_B = source_adapter_tensors[f"{base_name}.lora_B.weight"]

        if base_name in mamba_merge_bases:
            continue

        if ".experts.w3" in base_name and lora_A.numel() == 0:
            continue

        if candidate_spec["drop_lm_head"] and ".lm_head" in base_name:
            continue

        if ".experts.w1" in base_name or ".experts.w2" in base_name:
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()

            num_experts = lora_A.shape[0]
            for expert_idx in range(num_experts):
                expert_base = rewrite_expert_base_name(base_name, expert_idx)
                output_base = rename_base_name(
                    expert_base,
                    candidate_spec["namespace_mode"],
                )
                output_tensors[f"{output_base}.lora_A.weight"] = lora_A[expert_idx].contiguous()
                output_tensors[f"{output_base}.lora_B.weight"] = lora_B[expert_idx].contiguous()
            continue

        output_base = rename_base_name(
            base_name,
            candidate_spec["namespace_mode"],
        )
        output_tensors[f"{output_base}.lora_A.weight"] = lora_A.contiguous()
        output_tensors[f"{output_base}.lora_B.weight"] = lora_B.contiguous()

    for layer_prefix, components_map in sorted(mamba_merge_layers.items()):
        ordered_components = []
        for comp_name in ("gate_proj", "x_proj"):
            comp_base = components_map[comp_name]
            ordered_components.append(
                (
                    comp_name,
                    source_adapter_tensors[f"{comp_base}.lora_A.weight"],
                    source_adapter_tensors[f"{comp_base}.lora_B.weight"],
                )
            )

        if candidate_spec["strategy"] == "dense_svd":
            lora_A_parts = [item[1] for item in ordered_components]
            merged_lora_A = torch.cat(lora_A_parts, dim=0)

            row_offset = 0
            merged_rank = merged_lora_A.shape[0]
            fused_out_dim = sum(item[2].shape[0] for item in ordered_components)
            merged_lora_B = torch.zeros(
                fused_out_dim,
                merged_rank,
                dtype=merged_lora_A.dtype,
            )

            rank_offset = 0
            for _, _, lora_B in ordered_components:
                out_dim = lora_B.shape[0]
                rank = lora_B.shape[1]
                merged_lora_B[row_offset : row_offset + out_dim, rank_offset : rank_offset + rank] = lora_B
                row_offset += out_dim
                rank_offset += rank

            if merged_rank > FORCED_FUSED_RANK:
                merged_lora_B, merged_lora_A, stats = compress_dense_svd(
                    merged_lora_B,
                    merged_lora_A,
                    FORCED_FUSED_RANK,
                )
            else:
                stats = {
                    "rank": int(merged_rank),
                    "retained_energy": 1.0,
                    "retained_singular_mass": 1.0,
                }
        elif candidate_spec["strategy"] == "block_topk":
            merged_lora_B, merged_lora_A, stats = compress_block_topk(
                ordered_components,
                total_rank=FORCED_FUSED_RANK,
                min_rank_per_component=candidate_spec["min_rank_per_component"],
                component_priority=candidate_spec.get("component_priority", {}),
            )
        else:
            raise ValueError(f"Unknown strategy: {candidate_spec['strategy']}")

        output_base = rename_base_name(
            f"{layer_prefix}.in_proj",
            candidate_spec["namespace_mode"],
        )
        output_tensors[f"{output_base}.lora_A.weight"] = merged_lora_A
        output_tensors[f"{output_base}.lora_B.weight"] = merged_lora_B
        fused_stats.append(
            {
                "layer_prefix": layer_prefix,
                "strategy": candidate_spec["strategy"],
                **stats,
            }
        )

    return output_tensors, fused_stats


def save_candidate(candidate_spec: dict) -> dict:
    output_dir = OUTPUT_ROOT / candidate_spec["name"]
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_tensors, fused_stats = build_output_tensors(candidate_spec)
    adapter_config = build_adapter_config(output_tensors, candidate_spec)

    save_file(output_tensors, output_dir / "adapter_model.safetensors")
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
        f.write("\n")

    mean_retained_energy = (
        sum(item["retained_energy"] for item in fused_stats) / len(fused_stats)
        if fused_stats
        else 1.0
    )
    manifest = {
        **candidate_spec,
        "output_dir": str(output_dir),
        "tensor_count": len(output_tensors),
        "target_modules": adapter_config["target_modules"],
        "mean_fused_retained_energy": mean_retained_energy,
        "num_fused_layers": len(fused_stats),
    }

    with open(output_dir / "conversion_manifest.json", "w") as f:
        json.dump(
            {
                "manifest": manifest,
                "fused_layers": fused_stats,
            },
            f,
            indent=2,
        )
        f.write("\n")

    zip_base = WORK_DIR / f"submission_{candidate_spec['name']}"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", output_dir))
    manifest["zip_path"] = str(zip_path)
    return manifest
selected_specs = CANDIDATE_SPECS if BUILD_ALL_CANDIDATES else [ACTIVE_CANDIDATE]
candidate_manifests = []

for candidate_spec in selected_specs:
    print(f"Building candidate: {candidate_spec['name']}")
    candidate_manifests.append(save_candidate(candidate_spec))

candidate_manifests_df = pd.DataFrame(candidate_manifests).sort_values(
    ["mean_fused_retained_energy", "tensor_count"],
    ascending=[False, False],
)
candidate_manifests_df

if BUILD_ALL_CANDIDATES:
    selected_manifest = next(
        item for item in candidate_manifests if item["name"] == ACTIVE_CANDIDATE_NAME
    )
else:
    selected_manifest = candidate_manifests[0]

submission_zip = WORK_DIR / "submission.zip"
shutil.copy2(selected_manifest["zip_path"], submission_zip)

with open(WORK_DIR / "selected_candidate.json", "w") as f:
    json.dump(selected_manifest, f, indent=2)
    f.write("\n")

print("Active candidate:", selected_manifest["name"])
print("Submission zip:", submission_zip)
print("Candidate zip:", selected_manifest["zip_path"])
