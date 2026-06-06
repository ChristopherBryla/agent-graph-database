import uvicorn

from genesys.config import AppConfig
from genesys.component_manager.main_component_manager import ComponentManager
from genesys.dataprovider.main_data_provider import DataProvider
from genesys.hierarchy import Hierarchy
from genesys.storage.sqlite_store import SQLiteStore
from sample.ml_model.ml_sample_model1 import LinearRegression
from sample.storage.save_results import SaveResults

CONFIG_PATH = "src/genesys/component_manager/pipeline_config.yaml"

# All runtime settings come from the YAML — nothing hardcoded below this line.
cfg = AppConfig.from_yaml(CONFIG_PATH)


def make_components(entity_id: str) -> dict:
    """Component factory — called once per entity during a fan-out run.

    Returns fresh component instances configured for the given entity.
    Keeping this as a plain function makes it easy to swap components per
    entity or add entity-specific storage settings (e.g. pipeline_name).
    """
    return {
        "linear_regression": LinearRegression(),
        "save_results": SaveResults(
            store=SQLiteStore(db_path=cfg.db_path),
            pipeline_name=entity_id,
        ),
    }


# ---------------------------------------------------------------------------
# Define your hierarchy — fully agnostic, you name the levels
# ---------------------------------------------------------------------------
# The hierarchy is loaded from the same YAML config that defines the
# pipeline. The framework does not know or care what "account", "location",
# or "zone" mean — define as many or as few levels as your data model
# requires by editing the 'hierarchy' section of pipeline_config.yaml.

HIERARCHY = Hierarchy.from_yaml(CONFIG_PATH)


if __name__ == "__main__":
    provider = DataProvider()
    manager = ComponentManager.from_config(CONFIG_PATH, data_provider=provider)

    # ------------------------------------------------------------------
    # Option A — single entity run (baseline level from config)
    # ------------------------------------------------------------------
    baseline_entity = HIERARCHY.members_at(cfg.baseline_level)[0]
    for name, component in make_components(baseline_entity).items():
        manager.register(name, component)

    result = manager.run_pipeline(entity_id=baseline_entity)

    if result.succeeded:
        model = manager.get("linear_regression")
        prediction = model.predict(4.0, 5.0, 6.0, 7.0)
        print(f"[{baseline_entity}] sample prediction: {prediction:.4f}")

    # ------------------------------------------------------------------
    # Option B — fan-out at any level in the hierarchy
    # ------------------------------------------------------------------

    # Run for every location under the account:
    location_results = manager.run_pipeline_for_many(
        member_ids=HIERARCHY.members_at(cfg.baseline_level),
        component_factory=make_components,
        max_workers=cfg.fanout_max_workers,
    )

    # Run for every zone (one level below location):
    zone_results = manager.run_pipeline_for_many(
        member_ids=HIERARCHY.members_at("zone"),
        component_factory=make_components,
        max_workers=cfg.fanout_max_workers,
    )

    # Run for just the children of one specific entity:
    loc_001_zones = manager.run_pipeline_for_many(
        member_ids=HIERARCHY.children_of("loc_001"),
        component_factory=make_components,
        max_workers=cfg.fanout_max_workers,
    )

    for entity_id, run_result in location_results.items():
        status = "ok" if run_result.succeeded else "FAILED"
        print(f"[{entity_id}] {status}")

    # ------------------------------------------------------------------
    # Start the API server — all settings read from config
    # ------------------------------------------------------------------
    uvicorn.run(
        "genesys.app:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=cfg.server.reload,
    )
