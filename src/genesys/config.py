"""AppConfig — loads all runtime settings from pipeline_config.yaml.

This is the single object both the backend and (later) the API use to
read every configurable value. The frontend will modify the YAML; the
backend reads it here. No values should be hardcoded in app.py or main.py.

Usage::

    config = AppConfig.from_yaml("src/genesys/component_manager/pipeline_config.yaml")

    config.db_path            # "data/genesys.db"
    config.server_host        # "0.0.0.0"
    config.server_port        # 8000
    config.baseline_level     # "location"
    config.fanout_max_workers # None  (fully parallel)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


@dataclass
class AppConfig:
    # project identity
    project_name: str = "genesys"
    version: str = "0.1.0"

    # storage
    db_path: str = "data/genesys.db"
    blob_path: str = "data/blobs"

    # pipeline
    data_source: str = "sample"
    default_pipeline_name: str = "default"
    baseline_level: str = "location"

    # concurrency
    fanout_max_workers: int | None = None

    # server
    server: ServerConfig = field(default_factory=ServerConfig)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str) -> "AppConfig":
        """Read the 'settings' section of a pipeline_config.yaml file.

        Any key not present in the file falls back to the dataclass default,
        so the file only needs to contain values that differ from defaults.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        s = raw.get("settings", {})
        srv = s.get("server", {})

        return cls(
            project_name=s.get("project_name", "genesys"),
            version=str(s.get("version", "0.1.0")),
            db_path=s.get("db_path", "data/genesys.db"),
            blob_path=s.get("blob_path", "data/blobs"),
            data_source=s.get("data_source", "sample"),
            default_pipeline_name=s.get("default_pipeline_name", "default"),
            baseline_level=s.get("baseline_level", "location"),
            fanout_max_workers=s.get("fanout_max_workers", None),
            server=ServerConfig(
                host=srv.get("host", "0.0.0.0"),
                port=int(srv.get("port", 8000)),
                reload=bool(srv.get("reload", True)),
            ),
        )
