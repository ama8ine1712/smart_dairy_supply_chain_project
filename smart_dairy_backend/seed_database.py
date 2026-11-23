"""
Database Seeder for AI-Driven Smart Dairy Supply Chain Optimization
Creates all normalized tables in SQLite/PostgreSQL, populates master BMC hubs & tankers,
inserts 1,500 milk batch records, seeds initial IoT telemetry, runs VRPTW optimization,
and trains the Scikit-Learn models.
"""

from datetime import datetime, timedelta
import numpy as np
from app.database import engine, SessionLocal, Base
from app.models import (
    CollectionCenterBMC, RefrigeratedTanker, MilkBatchRecord, IoTTelemetryEvent
)
from app.ml_engine import train_models_from_db
from app.route_optimizer import solve_vrptw_routes


def seed_all(num_records: int = 1500):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Seed Master BMC Collection Centers
        bmc_specs = [
            ("BMC-Anand-North", "Anand", 22.5645, 72.9289, 28.0, 12000.0, 1.15),
            ("BMC-Mehsana-East", "Mehsana", 23.5880, 72.3693, 52.0, 15000.0, 1.25),
            ("BMC-Banaskantha-West", "Banaskantha", 24.1711, 72.4386, 118.0, 18000.0, 1.35),
            ("BMC-Sabarkantha-Central", "Sabarkantha", 23.8477, 72.9933, 76.0, 11000.0, 1.05),
            ("BMC-Kheda-South", "Kheda", 22.7507, 72.6847, 34.0, 10000.0, 0.95),
        ]
        if db.query(CollectionCenterBMC).count() == 0:
            for code, dist, lat, lon, base_d, cap, scale in bmc_specs:
                db.add(CollectionCenterBMC(
                    bmc_code=code,
                    region_district=dist,
                    latitude=lat,
                    longitude=lon,
                    base_distance_km=base_d,
                    chilling_capacity_liters=cap,
                    hub_scale_factor=scale,
                ))
            db.commit()

        # 2. Seed Master Refrigerated Tankers
        corridors = ["CORR-A1-Express", "CORR-B2-Rural", "CORR-C3-Highway", "CORR-D4-Feeder"]
        if db.query(RefrigeratedTanker).count() == 0:
            for i in range(1, 19):
                db.add(RefrigeratedTanker(
                    tanker_id=f"TKR-{i:02d}",
                    vehicle_registration=f"GJ-01-DX-{4100 + i}",
                    capacity_liters=10000.0 if i % 2 == 0 else 12000.0,
                    insulation_grade="PUF-HighDensity-100mm",
                    assigned_corridor=corridors[i % len(corridors)],
                    status="IN_TRANSIT" if i <= 8 else "ACTIVE",
                ))
            db.commit()

        # 3. Seed 1,500 Milk Batch Records
        if db.query(MilkBatchRecord).count() == 0:
            np.random.seed(42)
            start_date = datetime(2025, 1, 1, 5, 0)
            tanker_ids = [f"TKR-{i:02d}" for i in range(1, 19)]
            batch_objs = []

            for i in range(num_records):
                ts = start_date + timedelta(hours=12 * i + int(np.random.randint(0, 3)))
                dow = ts.weekday()
                month = ts.month
                if month in [11, 12, 1, 2]:
                    season = 1
                    ambient_temp = float(np.random.normal(22.5, 3.2))
                    seasonal_supply_mult = 1.18
                    seasonal_demand_mult = 1.05
                elif month in [3, 4, 5, 6]:
                    season = 2
                    ambient_temp = float(np.random.normal(38.5, 3.8))
                    seasonal_supply_mult = 0.84
                    seasonal_demand_mult = 1.22
                else:
                    season = 3
                    ambient_temp = float(np.random.normal(29.5, 2.8))
                    seasonal_supply_mult = 1.02
                    seasonal_demand_mult = 0.98

                bmc_idx = int(np.random.choice(len(bmc_specs), p=[0.24, 0.25, 0.22, 0.16, 0.13]))
                bmc_name, _, _, _, base_dist, _, hub_scale = bmc_specs[bmc_idx]
                corridor = corridors[bmc_idx % len(corridors)]
                tanker = str(np.random.choice(tanker_ids))

                weekend_boost = 1.08 if dow in [5, 6] else 1.0
                raw_vol = float(np.clip(np.random.normal(4800 * hub_scale * seasonal_supply_mult, 680), 1400, 9600))
                fat_pct = round(float(np.clip(np.random.normal(4.8 if season == 1 else 4.3, 0.55), 3.2, 6.9)), 2)
                snf_pct = round(float(np.clip(np.random.normal(8.65, 0.28), 8.0, 9.5)), 2)

                init_temp = round(float(np.clip(np.random.normal(3.1, 0.55), 2.0, 5.2)), 2)
                dist_km = round(float(np.clip(np.random.normal(base_dist, base_dist * 0.18), 12.0, 175.0)), 1)
                traffic_factor = float(np.random.uniform(0.022, 0.042))
                transit_hrs = round(dist_km * traffic_factor + float(np.random.uniform(0.25, 0.9)), 2)
                door_opens = int(np.clip(np.random.poisson(1.6 if dist_km < 60 else 2.8), 0, 8))

                insulation_drift = (ambient_temp - 20.0) * 0.045 + door_opens * 0.42 + transit_hrs * 0.28
                fault_spike = float(np.random.uniform(1.6, 3.6)) if (np.random.rand() < 0.14) else 0.0

                reefer_temp = round(float(np.clip(init_temp + insulation_drift + fault_spike + np.random.normal(0, 0.25), 2.0, 10.5)), 2)
                excursion_mins = int(np.clip(max(0.0, (reefer_temp - 4.3) * transit_hrs * 14.5 + np.random.normal(0, 5)), 0, 180))

                thermal_load = max(0.0, reefer_temp - 3.8) * (transit_hrs ** 1.15) + (excursion_mins / 45.0)
                milk_ph = round(float(np.clip(6.72 - 0.024 * thermal_load + np.random.normal(0, 0.025), 6.20, 6.84)), 2)
                mbrt_hrs = round(float(np.clip(5.6 - 0.34 * thermal_load + np.random.normal(0, 0.22), 0.8, 6.5)), 2)

                spoilage_score = (
                    0.55 * (reefer_temp - 4.6)
                    + 0.32 * (transit_hrs - 2.8)
                    + 0.018 * (excursion_mins - 25)
                    - 3.8 * (milk_ph - 6.54)
                    - 0.38 * (mbrt_hrs - 3.2)
                    + float(np.random.normal(0, 0.35))
                )
                spoilage_risk = 1 if spoilage_score > 0.25 else 0
                route_saved_km = round(float(np.clip(dist_km * np.random.uniform(0.08, 0.19), 1.2, 31.5)), 1)

                next_day_demand = round(float(
                    (raw_vol * 1.18 * weekend_boost * (seasonal_demand_mult / seasonal_supply_mult))
                    + (hub_scale * 950)
                    + (180 if dow == 4 else 0)
                    + np.random.normal(0, 260)
                ), 1)

                batch_objs.append(MilkBatchRecord(
                    batch_id=f"BATCH-{10001 + i}",
                    timestamp=ts,
                    collection_center_bmc=bmc_name,
                    route_corridor=corridor,
                    tanker_id=tanker,
                    day_of_week=dow,
                    season_index=season,
                    raw_milk_volume_liters=round(raw_vol, 1),
                    fat_content_pct=fat_pct,
                    snf_content_pct=snf_pct,
                    initial_chilling_temp_c=init_temp,
                    reefer_chamber_temp_c=reefer_temp,
                    ambient_external_temp_c=round(ambient_temp, 1),
                    transit_distance_km=dist_km,
                    transit_duration_hrs=transit_hrs,
                    reefer_door_open_count=door_opens,
                    thermal_excursion_mins=excursion_mins,
                    milk_ph_level=milk_ph,
                    mbrt_hours=mbrt_hrs,
                    optimal_route_saved_km=route_saved_km,
                    cold_chain_spoilage_risk=spoilage_risk,
                    next_day_demand_liters=next_day_demand,
                ))

            db.bulk_save_objects(batch_objs)
            db.commit()

        # 4. Seed Initial IoT Telemetry Logs
        if db.query(IoTTelemetryEvent).count() == 0:
            sample_events = [
                ("TKR-01", "BATCH-11496", 22.6120, 72.8910, 3.4, 31.2, 2350, "CLOSED", 24.3, False, None),
                ("TKR-03", "BATCH-11497", 23.6210, 72.4021, 5.8, 38.4, 2980, "CLOSED", 23.9, True, "NDDB Threshold Breach: Reefer chamber at 5.8°C (> 4.5°C)."),
                ("TKR-07", "BATCH-11498", 24.1120, 72.4910, 3.9, 36.1, 2500, "CLOSED", 24.1, False, None),
                ("TKR-12", "BATCH-11499", 23.8105, 72.9502, 4.9, 35.8, 2890, "OPEN", 23.7, True, "NDDB Threshold Breach: Reefer chamber at 4.9°C (> 4.5°C)."),
                ("TKR-15", "BATCH-11500", 22.8102, 72.6512, 3.1, 29.4, 2280, "CLOSED", 24.4, False, None),
            ]
            for tkr, bid, lat, lon, rtemp, atemp, rpm, door, volt, alert, msg in sample_events:
                db.add(IoTTelemetryEvent(
                    tanker_id=tkr, batch_id=bid, latitude=lat, longitude=lon,
                    reefer_temp_c=rtemp, ambient_temp_c=atemp, compressor_rpm=rpm,
                    door_status=door, battery_voltage=volt, alert_triggered=alert, alert_message=msg
                ))
            db.commit()

        # 5. Run initial VRPTW Route Optimization & ML Training
        solve_vrptw_routes(db)
        metrics = train_models_from_db(db)
        print(f"[SEED COMPLETE] Batches: {db.query(MilkBatchRecord).count()} | RF Acc: {metrics['rf_accuracy']} | GBR R2: {metrics['gbr_r2_score']}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_all(1500)
