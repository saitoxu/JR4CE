"""Generic preprocessing script for JR4CE / GLIT datasets.

This single script converts raw CSV files (``apply.csv``, ``users.csv``,
``jobs.csv``) into the knowledge-graph / interaction files consumed by JR4CE
(``train.txt``, ``val.txt``, ``test.txt``, ``kg.txt``, ``item_kg.txt``,
``info.txt`` and ``*_original_id_map.txt``).

Any subset of the supported attributes can be selected via ``--attributes``,
so the same script handles different attribute sets (e.g. the paper's GLIT-2021
used ``job_type,employment_type,industry`` and GLIT-2022 used
``job_type,employment_type,salary``) as well as arbitrary ablation settings.

The GLIT dataset itself cannot be released, but a small synthetic dataset is
bundled under ``preprocess/sample`` so the script can be run as-is::

    python preprocess/generate_data.py \
        --input preprocess/sample \
        --output preprocess/sample_out \
        --attributes job_type,employment_type,industry,salary

See ``preprocess/README.md`` for the expected CSV schema.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Attribute definitions
# ---------------------------------------------------------------------------
#
# Every attribute contributes three relations to the knowledge graph:
#   - item    : attribute value attached to a job (jobs.csv)
#   - hope    : the value a user *wants* (users.csv "hope_*" columns)
#   - current : the value a user *currently* has (users.csv "recent_*" columns)
#
# An attribute is therefore described by three ``Source`` objects, one per
# relation. A ``Source`` knows which CSV column(s) hold the categorical id(s)
# and how to read them.
#
# Every attribute value is expected to be a categorical id already present in
# the CSV. In particular the ``salary`` attribute assumes the income buckets
# have been assigned upstream (e.g. ``annual_income_id`` / ``recent_annual_income_id``);
# this script does not convert raw salaries into buckets.


# How to read a single relation source from a CSV row.
#   mode="single": take the value of the first column (if non-empty)
#   mode="multi" : dash-separated list of values (e.g. "1-2-3")
@dataclass(frozen=True)
class Source:
    columns: list[str]
    mode: str = "single"


@dataclass(frozen=True)
class Attribute:
    key: str
    # File name (without extension) of the new-id -> original-id mapping.
    map_name: str
    item: Source  # value on the job side (jobs.csv)
    hope: Source  # value the user wants (users.csv)
    current: Source  # value the user currently has (users.csv)


ATTRIBUTES: dict[str, Attribute] = {
    "job_type": Attribute(
        key="job_type",
        map_name="job_type_original_id_map",
        item=Source(["job_type_ids"], mode="multi"),
        hope=Source(["hope_job_type_ids"], mode="multi"),
        current=Source(["recent_job_type_id"], mode="single"),
    ),
    "employment_type": Attribute(
        key="employment_type",
        map_name="employment_type_original_id_map",
        item=Source(["employment_type_ids"], mode="multi"),
        hope=Source(["hope_employment_type_ids"], mode="multi"),
        current=Source(["employment_type_id"], mode="single"),
    ),
    "industry": Attribute(
        key="industry",
        map_name="industry_original_id_map",
        item=Source(["industry_id"], mode="single"),
        hope=Source(["hope_industry_ids"], mode="multi"),
        current=Source(["recent_industry_id"], mode="single"),
    ),
    "salary": Attribute(
        key="salary",
        map_name="annual_income_original_id_map",
        # All income values are expected as categorical bucket ids in the CSV.
        item=Source(["annual_income_id"], mode="single"),
        hope=Source(["hope_annual_income_id"], mode="single"),
        current=Source(["recent_annual_income_id"], mode="single"),
    ),
}


def extract_values(row: dict[str, str], source: Source) -> list[str]:
    """Read every non-empty categorical id of ``source`` from ``row``."""
    if source.mode == "multi":
        values: list[str] = []
        for column in source.columns:
            values.extend(v for v in row.get(column, "").split("-") if v)
        return values
    # "single"
    value = row.get(source.columns[0], "")
    return [value] if value else []


# ---------------------------------------------------------------------------
# CSV loading / filtering
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts, preserving row order."""
    rows: list[dict[str, str]] = []
    header: Optional[list[str]] = None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if header is None:
                header = row
            else:
                rows.append(dict(zip(header, row)))
    return rows


def filter_applies(
    applies: list[dict[str, str]],
    users: list[dict[str, str]],
    jobs: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Drop applies to unknown users/jobs and duplicate (user, job) pairs.

    Row order (chronological, old -> new) is preserved.
    """
    user_ids = {user["id"] for user in users}
    job_ids = {job["id"] for job in jobs}
    seen: set[tuple[str, str]] = set()
    filtered: list[dict[str, str]] = []
    for apply in applies:
        key = (apply["user_id"], apply["offer_id"])
        if apply["user_id"] not in user_ids or apply["offer_id"] not in job_ids:
            continue
        if key in seen:
            continue
        seen.add(key)
        filtered.append(apply)
    return filtered


def filter_jobs(
    jobs: list[dict[str, str]], applies: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Keep only jobs that received at least one apply."""
    job_ids = {apply["offer_id"] for apply in applies}
    return [job for job in jobs if job["id"] in job_ids]


def filter_users(
    users: list[dict[str, str]], applies: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Keep only users who made at least one apply."""
    user_ids = {apply["user_id"] for apply in applies}
    return [user for user in users if user["id"] in user_ids]


# ---------------------------------------------------------------------------
# Time-based train / val / test split
# ---------------------------------------------------------------------------
def split_applies(
    applies: list[dict[str, str]],
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Split applies per user using a time-based (leave-last-out) scheme.

    ``applies`` must be in chronological order (old -> new); ``offer_ids`` for
    each user therefore ends with the most recent application. The newest
    application becomes the test sample and the second newest becomes the
    validation sample, matching the paper's time-based protocol. Everything
    older is used for training.

    For users with exactly two applications we cannot afford one sample for
    each split, so the newest is assigned to test / val alternately (keeping
    the two evaluation sets balanced) and the older one is used for training.
    """
    user_applies: dict[str, list[str]] = {}
    for apply in applies:
        user_applies.setdefault(apply["user_id"], []).append(apply["offer_id"])

    train: list[list[str]] = []
    val: list[list[str]] = []
    test: list[list[str]] = []

    for i, (user_id, offer_ids) in enumerate(user_applies.items()):
        n = len(offer_ids)
        if n >= 3:
            test.append([user_id, offer_ids.pop()])  # newest -> test
            val.append([user_id, offer_ids.pop()])  # 2nd newest -> val
        elif n == 2:
            if i % 2 == 1:
                val.append([user_id, offer_ids.pop()])
            else:
                test.append([user_id, offer_ids.pop()])
        train += [[user_id, offer_id] for offer_id in offer_ids]
    return train, val, test


# ---------------------------------------------------------------------------
# Id assignment & knowledge-graph construction
# ---------------------------------------------------------------------------
def assign_new_ids(original_ids: list[str]) -> dict[str, int]:
    """Map each original id to a contiguous 0-based id (input order)."""
    return {original_id: i for i, original_id in enumerate(original_ids)}


@dataclass
class Edge:
    """A raw (not yet remapped) knowledge-graph edge.

    ``entity`` is the original attribute id; ``tail`` is the original job/user
    id; ``attr`` / ``side`` identify which relation the edge belongs to.
    """

    attr: str
    side: str  # "item" | "hope" | "current"
    entity: str
    tail: str  # original job id (item edges) or user id (user edges)


@dataclass
class Schema:
    """Relation ids and entity-id offsets derived from the selected attributes.

    Entity ids live in a single contiguous space laid out as::

        [ jobs | users | attr_0 values | attr_1 values | ... ]

    Relation ids are grouped by side so that ``info.txt`` lists the item /
    preference / current relations on three separate lines::

        item    relations : 0 .. k-1
        hope    relations : k .. 2k-1
        current relations : 2k .. 3k-1
    """

    item_relation: dict[str, int]
    hope_relation: dict[str, int]
    current_relation: dict[str, int]
    entity_offset: dict[str, int]
    entity_size: int


def build_schema(
    attributes: list[Attribute],
    attr_maps: dict[str, dict[str, int]],
    job_size: int,
    user_size: int,
) -> Schema:
    k = len(attributes)
    item_relation: dict[str, int] = {}
    hope_relation: dict[str, int] = {}
    current_relation: dict[str, int] = {}
    entity_offset: dict[str, int] = {}

    offset = job_size + user_size
    for i, attr in enumerate(attributes):
        item_relation[attr.key] = i
        hope_relation[attr.key] = k + i
        current_relation[attr.key] = 2 * k + i
        entity_offset[attr.key] = offset
        offset += len(attr_maps[attr.key])

    return Schema(
        item_relation=item_relation,
        hope_relation=hope_relation,
        current_relation=current_relation,
        entity_offset=entity_offset,
        entity_size=offset,
    )


def collect_edges(
    rows: list[dict[str, str]],
    attributes: list[Attribute],
    id_column: str,
    sides: list[str],
    attr_ids: dict[str, set[str]],
) -> list[Edge]:
    """Collect raw edges for the given ``sides`` and record attribute ids."""
    edges: list[Edge] = []
    for row in rows:
        tail = row[id_column]
        for attr in attributes:
            for side in sides:
                source = getattr(attr, side)
                for entity in extract_values(row, source):
                    edges.append(Edge(attr.key, side, entity, tail))
                    attr_ids[attr.key].add(entity)
    return edges


def remap_edges(
    edges: list[Edge],
    schema: Schema,
    attr_maps: dict[str, dict[str, int]],
    tail_map: dict[str, int],
    tail_offset: int,
) -> list[list[int]]:
    """Turn raw edges into ``[head_entity, relation, tail]`` triples."""
    relation_of = {
        "item": schema.item_relation,
        "hope": schema.hope_relation,
        "current": schema.current_relation,
    }
    triples: list[list[int]] = []
    for edge in edges:
        head = attr_maps[edge.attr][edge.entity] + schema.entity_offset[edge.attr]
        relation = relation_of[edge.side][edge.attr]
        tail = tail_map[edge.tail] + tail_offset
        triples.append([head, relation, tail])
    return triples


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def save_interaction(
    path: Path,
    interaction: list[list[str]],
    user_id_map: dict[str, int],
    job_id_map: dict[str, int],
) -> None:
    """Write ``user_id job_id job_id ...`` lines, sorted by id."""
    user_items: dict[int, set[int]] = {}
    for user_id, job_id in interaction:
        user_items.setdefault(user_id_map[user_id], set()).add(job_id_map[job_id])
    with open(path, "w") as f:
        for user_id in sorted(user_items):
            job_ids = " ".join(str(j) for j in sorted(user_items[user_id]))
            f.write(f"{user_id} {job_ids}\n")


def save_id_map(path: Path, mapper: dict[str, int]) -> None:
    """Write ``new_id original_id`` lines, sorted by new id."""
    reverse = {new_id: original_id for original_id, new_id in mapper.items()}
    with open(path, "w") as f:
        for new_id in sorted(reverse):
            f.write(f"{new_id} {reverse[new_id]}\n")


def save_graph(path: Path, graph: list[list[int]]) -> None:
    """Write ``head relation tail`` triples."""
    with open(path, "w") as f:
        for head, relation, tail in graph:
            f.write(f"{head} {relation} {tail}\n")


def save_info(
    path: Path,
    user_size: int,
    item_size: int,
    schema: Schema,
    attributes: list[Attribute],
) -> None:
    """Write ``info.txt`` consumed by JR4CE's dataset loader.

    Line 1: ``user_size item_size entity_size``
    Line 2: item relations    (one id per attribute)
    Line 3: preference (hope) relations
    Line 4: current relations
    """
    keys = [attr.key for attr in attributes]
    with open(path, "w") as f:
        f.write(f"{user_size} {item_size} {schema.entity_size}\n")
        f.write(" ".join(str(schema.item_relation[k]) for k in keys) + "\n")
        f.write(" ".join(str(schema.hope_relation[k]) for k in keys) + "\n")
        f.write(" ".join(str(schema.current_relation[k]) for k in keys) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate(input_dir: Path, output_dir: Path, attributes: list[Attribute]) -> None:
    users = load_csv(input_dir / "users.csv")
    jobs = load_csv(input_dir / "jobs.csv")
    applies = load_csv(input_dir / "apply.csv")

    applies = filter_applies(applies, users, jobs)
    jobs = filter_jobs(jobs, applies)
    users = filter_users(users, applies)

    train, val, test = split_applies(applies)

    user_id_map = assign_new_ids([user["id"] for user in users])
    job_id_map = assign_new_ids([job["id"] for job in jobs])
    job_size = len(job_id_map)
    user_size = len(user_id_map)

    # Collect raw attribute edges and the set of attribute ids per attribute.
    attr_ids: dict[str, set[str]] = {attr.key: set() for attr in attributes}
    item_edges = collect_edges(jobs, attributes, "id", ["item"], attr_ids)
    user_edges = collect_edges(users, attributes, "id", ["hope", "current"], attr_ids)

    attr_maps = {
        attr.key: assign_new_ids(sorted(attr_ids[attr.key])) for attr in attributes
    }
    schema = build_schema(attributes, attr_maps, job_size, user_size)

    # item edges have job tails (offset 0); user edges have user tails.
    item_triples = remap_edges(item_edges, schema, attr_maps, job_id_map, 0)
    user_triples = remap_edges(user_edges, schema, attr_maps, user_id_map, job_size)
    kg = item_triples + user_triples
    item_kg = item_triples

    # Reset the output directory.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    save_interaction(output_dir / "train.txt", train, user_id_map, job_id_map)
    save_interaction(output_dir / "val.txt", val, user_id_map, job_id_map)
    save_interaction(output_dir / "test.txt", test, user_id_map, job_id_map)

    save_id_map(output_dir / "user_original_id_map.txt", user_id_map)
    save_id_map(output_dir / "item_original_id_map.txt", job_id_map)
    for attr in attributes:
        save_id_map(output_dir / f"{attr.map_name}.txt", attr_maps[attr.key])

    save_graph(output_dir / "kg.txt", kg)
    save_graph(output_dir / "item_kg.txt", item_kg)
    save_info(output_dir / "info.txt", user_size, job_size, schema, attributes)

    print(
        f"Done. users={user_size} jobs={job_size} "
        f"entities={schema.entity_size} relations={3 * len(attributes)}\n"
        f"  train={len(train)} val={len(val)} test={len(test)}\n"
        f"  output -> {output_dir}"
    )


def parse_attributes(value: str) -> list[Attribute]:
    keys = [k.strip() for k in value.split(",") if k.strip()]
    if not keys:
        raise argparse.ArgumentTypeError("--attributes must not be empty")
    unknown = [k for k in keys if k not in ATTRIBUTES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown attribute(s): {', '.join(unknown)}. "
            f"choose from: {', '.join(ATTRIBUTES)}"
        )
    if len(set(keys)) != len(keys):
        raise argparse.ArgumentTypeError("--attributes must not contain duplicates")
    return [ATTRIBUTES[k] for k in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw CSV files into JR4CE dataset files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing apply.csv, users.csv and jobs.csv.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write the generated dataset files to.",
    )
    parser.add_argument(
        "--attributes",
        required=True,
        type=parse_attributes,
        help=(
            "Comma-separated attribute keys to include. "
            f"Available: {', '.join(ATTRIBUTES)}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.input, args.output, args.attributes)


if __name__ == "__main__":
    main()
