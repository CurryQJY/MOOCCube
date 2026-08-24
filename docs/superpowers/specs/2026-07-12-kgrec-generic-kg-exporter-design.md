# KGRec Generic KG Exporter Design

## Objective

Extend the strict KGRec data adapter so Junyi and COCO can be exported from
their canonical strict split pickles plus RecBole `.link` and `.kg` files.
The exporter must preserve arbitrary entity-to-entity KG triples, including
Junyi prerequisite edges, while keeping course items in the first `n_items`
entity positions required by KGRec.

## Data Contract

The exporter accepts:

- a strict split directory containing `static_train.pkl`, `static_val.pkl`,
  and `static_test.pkl` with `user_id` and `course_id` columns;
- a RecBole `.link` file with `item_id` and `entity_id` columns;
- a RecBole `.kg` file with `head_id`, `relation_id`, and `tail_id` columns;
- an output directory for KGRec atomic files.

Every split `course_id` must occur as an `entity_id` in `.link`. The split
course IDs define the item catalog; KG-only entities never become items.

## Entity and Relation Mapping

Course entities receive IDs `0..n_items-1` in deterministic sorted course-ID
order. All remaining entities found in either KG head or tail positions follow
in deterministic sorted order. Relations receive contiguous sorted IDs.

All KG triples are retained, including triples whose head and tail are both
non-course entities. Course KG degree counts appearances on either side so the
strict manifest can verify that every cold course has KG evidence.

## Output and Provenance

The exporter reuses the existing KGRec atomic writer and emits train,
validation, test, KG, mapping, and strict-manifest files. The manifest records
the split root, `.link` path, `.kg` path, full arbitrary-entity graph scope,
included relations, and relation edge counts.

The existing MOOCCube exporter remains unchanged.

## Validation and Errors

The exporter fails with a clear `ValueError` when a split course has no link
entity, when link item/entity mappings are duplicated inconsistently, or when
the required TSV columns are absent. Existing strict checks must pass:

- cold items absent from CF training;
- contiguous item, entity, and relation IDs;
- all cold items have at least one KG edge.

## Tests

Unit tests cover:

1. arbitrary external-to-external triples are retained;
2. course entities occupy the first `n_items` positions;
3. `.link` coverage is validated;
4. a temporary RecBole export produces a strict manifest with full relations;
5. existing MOOCCube adapter behavior remains green.

## Integration Verification

After unit tests pass, export Junyi and COCO seed 2025 from their canonical
strict split and RecBole KG files. Run full-model-configuration CUDA smokes
with `dim=64`, `context_hops=2`, `lr=1e-5`, and one training batch. These are
feasibility runs only; their metrics are not paper results.
