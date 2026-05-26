#!/usr/bin/env python3
"""Generate the Tribes mixed-radix action space schema.

The schema is the source of truth for the Python encoder and the Java capture
pipeline. This script derives the action type list from `src/core/Types.java`
and emits the JSON schema consumed by `ActionSpaceEncoder`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
TYPES_JAVA = ROOT / "src" / "core" / "Types.java"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "action_space_schema.json"

CANONICAL_ACTION_ORDER = [
    "END_TURN",
    "MOVE",
    "ATTACK",
    "CAPTURE",
    "CONVERT",
    "DISBAND",
    "EXAMINE",
    "HEAL_OTHERS",
    "MAKE_VETERAN",
    "RECOVER",
    "CLIMB_MOUNTAIN",
    "UPGRADE_BOAT",
    "UPGRADE_SHIP",
    "BUILD_ROAD",
    "SEND_STARS",
    "RESEARCH_TECH",
    "DECLARE_WAR",
    "BUILD",
    "SPAWN",
    "BURN_FOREST",
    "CLEAR_FOREST",
    "DESTROY",
    "GROW_FOREST",
    "LEVEL_UP",
    "RESOURCE_GATHERING",
]


def _extract_java_enum_members(java_text: str, enum_name: str) -> List[str]:
    pattern = rf"public enum {re.escape(enum_name)}\s*\{{(.*?)\n\s*private "
    match = re.search(pattern, java_text, re.S)
    if match is None:
        raise ValueError(f"Could not find enum {enum_name} in Types.java")

    members: List[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        line = line.split("//", 1)[0].strip().rstrip(",")
        if not line:
            continue
        name = line.split("(", 1)[0].strip()
        if name:
            members.append(name)
    return members


def _build_action_type_values(action_types: List[str], total_size: int = 32) -> List[str]:
    values = list(action_types)
    while len(values) < total_size:
        values.append(f"_PADDING_{len(values)}")
    return values


def build_schema(board_size: int = 11) -> Dict[str, Any]:
    with open(TYPES_JAVA, "r", encoding="utf-8") as handle:
        java_text = handle.read()

    action_types = _extract_java_enum_members(java_text, "ACTION")
    building_types = _extract_java_enum_members(java_text, "BUILDING")
    unit_types = _extract_java_enum_members(java_text, "UNIT")
    technology_types = _extract_java_enum_members(java_text, "TECHNOLOGY")

    if set(action_types) != set(CANONICAL_ACTION_ORDER):
        missing = sorted(set(CANONICAL_ACTION_ORDER) - set(action_types))
        extra = sorted(set(action_types) - set(CANONICAL_ACTION_ORDER))
        raise ValueError(
            "ACTION enum and canonical schema order diverged: "
            f"missing={missing}, extra={extra}"
        )

    action_type_values = _build_action_type_values(CANONICAL_ACTION_ORDER, total_size=32)
    action_type_index_map = {name: idx for idx, name in enumerate(CANONICAL_ACTION_ORDER)}
    tribe_count = 12
    max_units = 100
    max_cities = 50
    target_actor_size = 1 + board_size * board_size + max_units + max_cities + tribe_count

    schema = {
        "name": "Tribes Factorized Action Space",
        "board_size": board_size,
        "max_units": max_units,
        "max_cities": max_cities,
        "max_tribes": tribe_count,
        "version": "1.0",
        "description": (
            "Mixed-radix factorized action encoding for Tribes. NN outputs separate logits "
            "for [action_type, source_actor, target_actor, param], which are composed with masking "
            "to produce action distributions."
        ),
        "components": {
            "action_type": {
                "size": 32,
                "description": "Action type",
                "values": action_type_values,
                "index_map": action_type_index_map,
            },
            "source_actor": {
                "size": 151,
                "description": "Source actor (unit or city). Index 0 = None. Units: 1-100. Cities: 101-150.",
                "ranges": {
                    "none": 0,
                    "units": [1, 100],
                    "cities": [101, 150],
                },
                "encoding": "none=0, unit_id in [1..100], city_id + 100 in [101..150]",
            },
            "target_actor": {
                "size": target_actor_size,
                "description": "Target actor or position. Indices: 0=None, 1-121=board positions (x*11+y), 122-221=unit IDs, 222-271=city IDs, 272-283=tribe IDs.",
                "ranges": {
                    "none": 0,
                    "board_positions": [1, 121],
                    "unit_ids": [122, 221],
                    "city_ids": [222, 271],
                    "tribe_ids": [272, 283],
                },
                "encoding": "none=0, pos as (x*board_size+y)+1 in [1..121], unit_id+121 in [122..221], city_id+221 in [222..271], tribe_id+271 in [272..283]",
            },
            "param": {
                "size": 80,
                "description": "Numeric/enum parameter. Content depends on action type. Indices 0-79 are shared; meaning determined by action context.",
                "sub_encodings": {
                    "building_type": f"0-{len(building_types) - 1} ({len(building_types)} buildings: {', '.join(building_types[:4])}, ...)",
                    "unit_type": f"0-{len(unit_types) - 1} ({len(unit_types)} units: {', '.join(unit_types[:4])}, ...)",
                    "technology": f"0-{len(technology_types) - 1} ({len(technology_types)} techs: {', '.join(technology_types[:4])}, ...)",
                    "num_stars": "0-100 (send 0 to 100 stars, or pick 0 if not applicable)",
                    "city_level_up_choice": "0-7 (WORKSHOP, EXPLORER, CITY_WALL, RESOURCES, POP_GROWTH, BORDER_GROWTH, PARK, SUPERUNIT)",
                    "tribe_id": f"0-{tribe_count - 1} ({tribe_count} tribes for DECLARE_WAR, SEND_STARS)",
                },
            },
        },
        "action_signatures": {
            "END_TURN": {
                "components": ["action_type"],
                "description": "End current tribe's turn. No parameters.",
                "example_encoding": {"action_type": 0, "source_actor": 0, "target_actor": 0, "param": 0},
            },
            "MOVE": {
                "components": ["action_type", "source_actor", "target_actor"],
                "source_type": "unit",
                "target_type": "board_position",
                "description": "Move unit from current position to target position.",
                "example_encoding": {"action_type": 1, "source_actor": 5, "target_actor": 45, "param": 0},
            },
            "ATTACK": {
                "components": ["action_type", "source_actor", "target_actor"],
                "source_type": "unit",
                "target_type": "board_position_or_unit",
                "description": "Attack target position/unit from source unit.",
                "example_encoding": {"action_type": 2, "source_actor": 5, "target_actor": 50, "param": 0},
            },
            "BUILD_ROAD": {
                "components": ["action_type", "target_actor"],
                "source_type": "implicit_tribe",
                "target_type": "board_position",
                "description": "Build road at target position (tribe-wide action).",
                "example_encoding": {"action_type": 13, "source_actor": 0, "target_actor": 30, "param": 0},
            },
            "SEND_STARS": {
                "components": ["action_type", "target_actor", "param"],
                "source_type": "implicit_tribe",
                "target_type": "tribe",
                "param_type": "num_stars",
                "description": "Send stars to target tribe (amount in param).",
                "example_encoding": {"action_type": 14, "source_actor": 0, "target_actor": 275, "param": 10},
            },
            "RESEARCH_TECH": {
                "components": ["action_type", "param"],
                "source_type": "implicit_tribe",
                "param_type": "technology",
                "description": "Research technology (tech index in param).",
                "example_encoding": {"action_type": 15, "source_actor": 0, "target_actor": 0, "param": 5},
            },
            "DECLARE_WAR": {
                "components": ["action_type", "target_actor"],
                "source_type": "implicit_tribe",
                "target_type": "tribe",
                "description": "Declare war on target tribe.",
                "example_encoding": {"action_type": 16, "source_actor": 0, "target_actor": 273, "param": 0},
            },
            "BUILD": {
                "components": ["action_type", "source_actor", "target_actor", "param"],
                "source_type": "city",
                "target_type": "board_position",
                "param_type": "building_type",
                "description": "Build building at target position from source city.",
                "example_encoding": {"action_type": 17, "source_actor": 105, "target_actor": 40, "param": 3},
            },
            "SPAWN": {
                "components": ["action_type", "source_actor", "param"],
                "source_type": "city",
                "param_type": "unit_type",
                "description": "Spawn unit type from source city.",
                "example_encoding": {"action_type": 18, "source_actor": 105, "target_actor": 0, "param": 0},
            },
            "BURN_FOREST": {
                "components": ["action_type", "source_actor", "target_actor"],
                "source_type": "city",
                "target_type": "board_position",
                "description": "Burn forest at target position (city action).",
                "example_encoding": {"action_type": 19, "source_actor": 105, "target_actor": 50, "param": 0},
            },
            "CLEAR_FOREST": {
                "components": ["action_type", "source_actor", "target_actor"],
                "source_type": "city",
                "target_type": "board_position",
                "description": "Clear forest at target position.",
                "example_encoding": {"action_type": 20, "source_actor": 105, "target_actor": 50, "param": 0},
            },
            "DESTROY": {
                "components": ["action_type", "source_actor", "target_actor"],
                "source_type": "city",
                "target_type": "board_position",
                "description": "Destroy at target position.",
                "example_encoding": {"action_type": 21, "source_actor": 105, "target_actor": 50, "param": 0},
            },
            "GROW_FOREST": {
                "components": ["action_type", "source_actor", "target_actor"],
                "source_type": "city",
                "target_type": "board_position",
                "description": "Grow forest at target position.",
                "example_encoding": {"action_type": 22, "source_actor": 105, "target_actor": 50, "param": 0},
            },
            "LEVEL_UP": {
                "components": ["action_type", "source_actor", "param"],
                "source_type": "city",
                "param_type": "city_level_up_choice",
                "description": "Level up city (choice in param).",
                "example_encoding": {"action_type": 23, "source_actor": 105, "target_actor": 0, "param": 0},
            },
            "RESOURCE_GATHERING": {
                "components": ["action_type", "source_actor"],
                "source_type": "city",
                "description": "Gather resources from city location.",
                "example_encoding": {"action_type": 24, "source_actor": 105, "target_actor": 0, "param": 0},
            },
        },
        "encoding_helpers": {
            "position_to_index": "x * board_size + y (e.g., x=3, y=5 -> 3*11+5=38, then add 1 -> index 39 in target_actor)",
            "unit_id_to_source_index": "unit_id (1-indexed in source_actor), or unit_id + 121 in target_actor for targeting units",
            "city_id_to_source_index": "city_id + 100 (101-150 in source_actor), or city_id + 221 in target_actor for targeting cities",
            "tribe_id_to_target_index": "tribe_id + 271 (272-283 in target_actor)",
        },
    }
    return schema


def write_schema(output_path: Path) -> None:
    schema = build_schema()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate action_space_schema.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_PATH, help="Schema output path")
    args = parser.parse_args()
    write_schema(args.output)
    print(f"Wrote schema to {args.output}")


if __name__ == "__main__":
    main()
