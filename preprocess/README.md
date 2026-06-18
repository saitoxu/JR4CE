# Dataset preprocessing

`generate_data.py` converts raw CSV files into the dataset files that JR4CE
consumes. It is a single, general-purpose script: any subset of the supported
attributes can be selected with `--attributes`, so the same script handles
different attribute sets and arbitrary ablation settings.

> **Note on the source data.** The GLIT dataset used in the paper cannot be
> released, and the SQL used to extract the raw CSVs from our internal database
> is specific to a proprietary schema, so it is omitted as well. Prepare the
> CSV files described below from your own data source. For reference, the paper
> used these attribute sets:
>
> - **GLIT-2021**: `job_type,employment_type,industry`
> - **GLIT-2022**: `job_type,employment_type,salary`

## Usage

A small synthetic dataset is bundled under [`sample/`](sample) so the script
can be run as-is:

```bash
# Run on the bundled sample data (all four attributes)
python preprocess/generate_data.py \
    --input preprocess/sample \
    --output preprocess/sample_out \
    --attributes job_type,employment_type,industry,salary

# Use any subset of attributes, e.g. an ablation that drops employment_type
python preprocess/generate_data.py \
    --input preprocess/sample \
    --output preprocess/sample_out \
    --attributes job_type,industry,salary
```

### Arguments

| Argument | Description |
| -- | -- |
| `--input` | Directory containing `apply.csv`, `users.csv`, `jobs.csv`. |
| `--output` | Directory to write the generated files to (created/overwritten). |
| `--attributes` | Comma-separated attribute keys to include (see below). |

Supported attribute keys: `job_type`, `employment_type`, `industry`, `salary`.

## Input CSV schema

Place the three files below in the `--input` directory. Only the columns
required by the attributes you select need to be present (plus the required
columns).

### `apply.csv`

The application log. **Rows must be sorted in chronological order
(oldest → newest)**: the split uses each user's most recent application as the
test sample and the next most recent as the validation sample.

| Column | Required | Description |
| -- | -- | -- |
| `user_id` | ✅ | Applicant id (matches `users.csv` `id`). |
| `offer_id` | ✅ | Job id (matches `jobs.csv` `id`). |
| `timestamp` | – | Not read directly; only used to sort the rows beforehand. |

### `users.csv`

| Column | Required | Used by | Description |
| -- | -- | -- | -- |
| `id` | ✅ | – | User id. |
| `hope_job_type_ids` | | `job_type` | Desired job types, dash-separated (`1-2-3`). |
| `recent_job_type_id` | | `job_type` | Current/most recent job type. |
| `hope_employment_type_ids` | | `employment_type` | Desired employment types, dash-separated. |
| `employment_type_id` | | `employment_type` | Current employment type. |
| `hope_industry_ids` | | `industry` | Desired industries, dash-separated. |
| `recent_industry_id` | | `industry` | Current/most recent industry. |
| `hope_annual_income_id` | | `salary` | Desired income as a category id. |
| `recent_annual_income_id` | | `salary` | Current income as a category id. |

### `jobs.csv`

| Column | Required | Used by | Description |
| -- | -- | -- | -- |
| `id` | ✅ | – | Job id. |
| `job_type_ids` | | `job_type` | Job types, dash-separated. |
| `employment_type_ids` | | `employment_type` | Employment types, dash-separated. |
| `industry_id` | | `industry` | Industry. |
| `annual_income_id` | | `salary` | Income as a category id. |

#### Note on categorical values

Every attribute value (including `salary`) is expected to be a **categorical
id** already present in the CSV. In particular, raw salaries must be bucketed
into income categories upstream (e.g. during data extraction); `generate_data.py`
does not convert raw amounts into buckets.

## Output files

All files are written under `--output`.

| File | Format (per line) | Description |
| -- | -- | -- |
| `train.txt` / `val.txt` / `test.txt` | `user_id job_id [job_id ...]` | Interactions, ids reindexed from 0; sorted by id. |
| `kg.txt` | `head_entity relation tail` | Full knowledge graph (item + user edges). |
| `item_kg.txt` | `head_entity relation tail` | Item-side edges only. |
| `info.txt` | see below | Sizes and relation ids used by JR4CE. |
| `user_original_id_map.txt` | `new_id original_id` | User id mapping. |
| `item_original_id_map.txt` | `new_id original_id` | Job id mapping. |
| `<attribute>_original_id_map.txt` | `new_id original_id` | One per selected attribute (`salary` → `annual_income_original_id_map.txt`). |

### `info.txt`

```
<user_size> <item_size> <entity_size>
<item relations>        # one id per attribute
<preference relations>  # the user's "hope_*" relations
<current relations>     # the user's "recent_*" relations
```

Entity ids form one contiguous space laid out as
`[ jobs | users | attribute_0 values | attribute_1 values | ... ]`, and
relations are numbered dynamically from the selected attributes: with `k`
attributes the item relations are `0..k-1`, the preference relations
`k..2k-1` and the current relations `2k..3k-1` (so the order/number of
relations follows whatever `--attributes` you pass).

## Sample data

[`sample/`](sample) contains a tiny synthetic dataset (6 users, 5 jobs) with
all columns required by the four attributes, so you can run `generate_data.py` and
inspect its output without any real data. It is **not** the dataset used in the
paper; it only illustrates the expected CSV format.
