-- Portable Relational Schema for Smart Dairy Supply Chain Optimization (PostgreSQL / SQLite)

CREATE TABLE IF NOT EXISTS collection_centers_bmc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bmc_code VARCHAR(64) NOT NULL UNIQUE,
    region_district VARCHAR(64) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    base_distance_km DOUBLE PRECISION NOT NULL,
    chilling_capacity_liters DOUBLE PRECISION NOT NULL,
    hub_scale_factor DOUBLE PRECISION DEFAULT 1.0,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS refrigerated_tankers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tanker_id VARCHAR(32) NOT NULL UNIQUE,
    vehicle_registration VARCHAR(32) NOT NULL UNIQUE,
    capacity_liters DOUBLE PRECISION NOT NULL DEFAULT 10000.0,
    insulation_grade VARCHAR(32) DEFAULT 'PUF-HighDensity',
    assigned_corridor VARCHAR(64) NOT NULL,
    status VARCHAR(32) DEFAULT 'ACTIVE',
    last_calibration_date VARCHAR(32) DEFAULT '2026-10-15'
);

CREATE TABLE IF NOT EXISTS milk_batch_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id VARCHAR(32) NOT NULL UNIQUE,
    timestamp TIMESTAMP NOT NULL,
    collection_center_bmc VARCHAR(64) NOT NULL REFERENCES collection_centers_bmc(bmc_code),
    route_corridor VARCHAR(64) NOT NULL,
    tanker_id VARCHAR(32) NOT NULL REFERENCES refrigerated_tankers(tanker_id),
    day_of_week INTEGER NOT NULL,
    season_index INTEGER NOT NULL,
    raw_milk_volume_liters DOUBLE PRECISION NOT NULL,
    fat_content_pct DOUBLE PRECISION NOT NULL,
    snf_content_pct DOUBLE PRECISION NOT NULL,
    initial_chilling_temp_c DOUBLE PRECISION NOT NULL,
    reefer_chamber_temp_c DOUBLE PRECISION NOT NULL,
    ambient_external_temp_c DOUBLE PRECISION NOT NULL,
    transit_distance_km DOUBLE PRECISION NOT NULL,
    transit_duration_hrs DOUBLE PRECISION NOT NULL,
    reefer_door_open_count INTEGER NOT NULL DEFAULT 0,
    thermal_excursion_mins INTEGER NOT NULL DEFAULT 0,
    milk_ph_level DOUBLE PRECISION NOT NULL,
    mbrt_hours DOUBLE PRECISION NOT NULL,
    optimal_route_saved_km DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cold_chain_spoilage_risk INTEGER NOT NULL,
    next_day_demand_liters DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bmc_timestamp ON milk_batch_records(collection_center_bmc, timestamp);
CREATE INDEX IF NOT EXISTS idx_spoilage_season ON milk_batch_records(cold_chain_spoilage_risk, season_index);

CREATE TABLE IF NOT EXISTS iot_telemetry_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tanker_id VARCHAR(32) NOT NULL REFERENCES refrigerated_tankers(tanker_id),
    batch_id VARCHAR(32),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    reefer_temp_c DOUBLE PRECISION NOT NULL,
    ambient_temp_c DOUBLE PRECISION NOT NULL,
    compressor_rpm INTEGER DEFAULT 2400,
    door_status VARCHAR(16) DEFAULT 'CLOSED',
    battery_voltage DOUBLE PRECISION DEFAULT 24.2,
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_message VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS optimized_route_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_code VARCHAR(48) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    depot_name VARCHAR(64) NOT NULL,
    total_stops INTEGER NOT NULL,
    total_volume_liters DOUBLE PRECISION NOT NULL,
    baseline_distance_km DOUBLE PRECISION NOT NULL,
    optimized_distance_km DOUBLE PRECISION NOT NULL,
    distance_saved_km DOUBLE PRECISION NOT NULL,
    distance_reduction_pct DOUBLE PRECISION NOT NULL,
    fuel_saved_liters DOUBLE PRECISION NOT NULL,
    co2_reduced_kg DOUBLE PRECISION NOT NULL,
    route_sequence_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    rf_accuracy DOUBLE PRECISION NOT NULL,
    rf_f1_score DOUBLE PRECISION NOT NULL,
    rf_roc_auc DOUBLE PRECISION NOT NULL,
    logistic_accuracy DOUBLE PRECISION NOT NULL,
    gbr_r2_score DOUBLE PRECISION NOT NULL,
    gbr_mae_liters DOUBLE PRECISION NOT NULL,
    ridge_r2_score DOUBLE PRECISION NOT NULL,
    artifact_path VARCHAR(255) NOT NULL
);
