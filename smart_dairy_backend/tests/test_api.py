from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_and_counts():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["counts"]["milk_batch_records"] >= 1500
    assert data["counts"]["bmc_hubs"] == 5
    assert data["counts"]["refrigerated_tankers"] == 18


def test_kpis_endpoint():
    res = client.get("/api/dashboard/kpis")
    assert res.status_code == 200
    data = res.json()
    assert data["total_batches"] >= 1500
    assert len(data["hub_summary"]) == 5


def test_paginated_batches():
    res = client.get("/api/batches?page=1&page_size=20&bmc=BMC-Anand-North")
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 20
    assert all(item["collection_center_bmc"] == "BMC-Anand-North" for item in data["items"])


def test_ml_prediction():
    payload = {
        "collection_center_bmc": "BMC-Banaskantha-West",
        "day_of_week": 4,
        "season_index": 2,
        "raw_milk_volume_liters": 6200.0,
        "fat_content_pct": 4.4,
        "snf_content_pct": 8.65,
        "initial_chilling_temp_c": 3.2,
        "reefer_chamber_temp_c": 6.4,
        "ambient_external_temp_c": 39.0,
        "transit_distance_km": 120.0,
        "transit_duration_hrs": 3.8,
        "reefer_door_open_count": 3,
        "thermal_excursion_mins": 75,
        "milk_ph_level": 6.42,
        "mbrt_hours": 1.9,
    }
    res = client.post("/api/ml/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["spoilage_probability_rf"] > 0.7
    assert data["spoilage_risk_label"] == "CRITICAL_SPOILAGE_RISK"


def test_vrptw_route_optimizer():
    res = client.post("/api/routes/optimize", json={"tanker_capacity_liters": 15000.0})
    assert res.status_code == 200
    data = res.json()
    assert data["distance_saved_km"] > 0
    assert data["total_stops"] == 5
