from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Text, Index
)
from sqlalchemy.orm import relationship
from app.database import Base


class CollectionCenterBMC(Base):
    """Master table of Bulk Milk Cooler (BMC) collection hubs."""
    __tablename__ = "collection_centers_bmc"

    id = Column(Integer, primary_key=True, index=True)
    bmc_code = Column(String(64), unique=True, nullable=False, index=True)
    region_district = Column(String(64), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    base_distance_km = Column(Float, nullable=False)
    chilling_capacity_liters = Column(Float, nullable=False)
    hub_scale_factor = Column(Float, default=1.0)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    batches = relationship("MilkBatchRecord", back_populates="bmc_hub")


class RefrigeratedTanker(Base):
    """Master table of insulated IoT-enabled milk transport tankers."""
    __tablename__ = "refrigerated_tankers"

    id = Column(Integer, primary_key=True, index=True)
    tanker_id = Column(String(32), unique=True, nullable=False, index=True)
    vehicle_registration = Column(String(32), unique=True, nullable=False)
    capacity_liters = Column(Float, nullable=False, default=10000.0)
    insulation_grade = Column(String(32), default="PUF-HighDensity")
    assigned_corridor = Column(String(64), nullable=False)
    status = Column(String(32), default="ACTIVE")  # ACTIVE, IN_TRANSIT, MAINTENANCE
    last_calibration_date = Column(String(32), default="2026-10-15")

    batches = relationship("MilkBatchRecord", back_populates="tanker")
    telemetry_events = relationship("IoTTelemetryEvent", back_populates="tanker")


class MilkBatchRecord(Base):
    """Core fact table storing milk procurement, cold-chain, quality, and demand records."""
    __tablename__ = "milk_batch_records"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(32), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    collection_center_bmc = Column(String(64), ForeignKey("collection_centers_bmc.bmc_code"), nullable=False, index=True)
    route_corridor = Column(String(64), nullable=False, index=True)
    tanker_id = Column(String(32), ForeignKey("refrigerated_tankers.tanker_id"), nullable=False, index=True)

    day_of_week = Column(Integer, nullable=False)
    season_index = Column(Integer, nullable=False)  # 1=Winter Flush, 2=Summer Lean, 3=Monsoon

    raw_milk_volume_liters = Column(Float, nullable=False)
    fat_content_pct = Column(Float, nullable=False)
    snf_content_pct = Column(Float, nullable=False)

    initial_chilling_temp_c = Column(Float, nullable=False)
    reefer_chamber_temp_c = Column(Float, nullable=False)
    ambient_external_temp_c = Column(Float, nullable=False)

    transit_distance_km = Column(Float, nullable=False)
    transit_duration_hrs = Column(Float, nullable=False)
    reefer_door_open_count = Column(Integer, nullable=False, default=0)
    thermal_excursion_mins = Column(Integer, nullable=False, default=0)

    milk_ph_level = Column(Float, nullable=False)
    mbrt_hours = Column(Float, nullable=False)
    optimal_route_saved_km = Column(Float, nullable=False, default=0.0)

    cold_chain_spoilage_risk = Column(Integer, nullable=False, index=True)  # 0=Safe, 1=At Risk
    next_day_demand_liters = Column(Float, nullable=False)

    bmc_hub = relationship("CollectionCenterBMC", back_populates="batches")
    tanker = relationship("RefrigeratedTanker", back_populates="batches")

    __table_args__ = (
        Index("idx_bmc_timestamp", "collection_center_bmc", "timestamp"),
        Index("idx_spoilage_season", "cold_chain_spoilage_risk", "season_index"),
    )


class IoTTelemetryEvent(Base):
    """High-frequency cold-chain sensor readings streamed from tanker edge gateways."""
    __tablename__ = "iot_telemetry_events"

    id = Column(Integer, primary_key=True, index=True)
    tanker_id = Column(String(32), ForeignKey("refrigerated_tankers.tanker_id"), nullable=False, index=True)
    batch_id = Column(String(32), nullable=True, index=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    reefer_temp_c = Column(Float, nullable=False)
    ambient_temp_c = Column(Float, nullable=False)
    compressor_rpm = Column(Integer, default=2400)
    door_status = Column(String(16), default="CLOSED")  # CLOSED, OPEN
    battery_voltage = Column(Float, default=24.2)
    alert_triggered = Column(Boolean, default=False)
    alert_message = Column(String(255), nullable=True)

    tanker = relationship("RefrigeratedTanker", back_populates="telemetry_events")


class OptimizedRoutePlan(Base):
    """Persisted VRPTW route optimization runs comparing baseline vs. AI routing."""
    __tablename__ = "optimized_route_plans"

    id = Column(Integer, primary_key=True, index=True)
    plan_code = Column(String(48), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    depot_name = Column(String(64), default="Mother Dairy Processing Plant - Gandhinagar")
    total_stops = Column(Integer, nullable=False)
    total_volume_liters = Column(Float, nullable=False)
    baseline_distance_km = Column(Float, nullable=False)
    optimized_distance_km = Column(Float, nullable=False)
    distance_saved_km = Column(Float, nullable=False)
    distance_reduction_pct = Column(Float, nullable=False)
    fuel_saved_liters = Column(Float, nullable=False)
    co2_reduced_kg = Column(Float, nullable=False)
    route_sequence_json = Column(Text, nullable=False)


class MLModelRun(Base):
    """Audit trail of ML training sessions and validation metrics."""
    __tablename__ = "ml_model_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_timestamp = Column(DateTime, default=datetime.utcnow)
    training_rows = Column(Integer, nullable=False)
    test_rows = Column(Integer, nullable=False)
    rf_accuracy = Column(Float, nullable=False)
    rf_f1_score = Column(Float, nullable=False)
    rf_roc_auc = Column(Float, nullable=False)
    logistic_accuracy = Column(Float, nullable=False)
    gbr_r2_score = Column(Float, nullable=False)
    gbr_mae_liters = Column(Float, nullable=False)
    ridge_r2_score = Column(Float, nullable=False)
    artifact_path = Column(String(255), nullable=False)
