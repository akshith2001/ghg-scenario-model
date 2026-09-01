"""Fetch the locked EPA ArcGIS extract used by the real-data benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "https://geopub.epa.gov/ArcGIS/rest/services/myenv/myenvlayers/MapServer/5/query"
FIELDS = [
    "YEAR",
    "ORISPL",
    "PNAME",
    "PSTATABB",
    "PLPRMFL",
    "PLHTIAN",
    "PLNGENAN",
    "NAMEPCAP",
    "CAPFAC",
    "PLCO2AN",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parameters = {
        "where": "PLHTIAN IS NOT NULL AND PLCO2AN IS NOT NULL",
        "outFields": ",".join(FIELDS),
        "returnGeometry": "false",
        "resultRecordCount": "32000",
        "orderByFields": "ORISPL",
        "f": "json",
    }
    url = f"{BASE_URL}?{urlencode(parameters)}"
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(f"EPA ArcGIS query failed: {payload['error']}")

    raw_path = Path("data/raw/egrid_natural_gas_2018.json")
    csv_path = Path("data/egrid_natural_gas_2018.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = [feature["attributes"] for feature in payload["features"]]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "query_url": url,
                "records": len(rows),
                "raw_sha256": sha256(raw_path),
                "processed_sha256": sha256(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
