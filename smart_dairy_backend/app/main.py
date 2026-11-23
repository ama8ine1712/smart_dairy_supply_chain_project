import io
import csv
import asyncio
import random
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import settings
from app.database import engine, get_db, Base
from app.models import (
    CollectionCenterBMC, RefrigeratedTanker, MilkBatchRecord,
    IoTTelemetryEvent, OptimizedRoutePlan, MLModelRun
)
from app.schemas import (
    BMCHubOut, TankerOut, MilkBatchCreate, MilkBatchOut,
    PaginatedBatchesResponse, IoTTelemetryIngest, IoTTelemetryOut,
    PredictionRequest, PredictionResponse, RouteOptimizeRequest, RouteOptimizeResponse
)
from app.ml_engine import train_models_from_db, predict_batch_risk_and_demand, load_or_train_bundle
from app.route_optimizer import solve_vrptw_routes

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise FastAPI + Relational Database + ML + VRPTW Backend for Smart Dairy Supply Chain Optimization.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    total_batches = db.query(func.count(MilkBatchRecord.id)).scalar()
    total_hubs = db.query(func.count(CollectionCenterBMC.id)).scalar()
    total_tankers = db.query(func.count(RefrigeratedTanker.id)).scalar()
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database_url": settings.DATABASE_URL.split("@")[-1],
        "counts": {
            "milk_batch_records": total_batches,
            "bmc_hubs": total_hubs,
            "refrigerated_tankers": total_tankers,
        },
    }


@app.get("/api/dashboard/kpis", tags=["Analytics"])
def get_dashboard_kpis(db: Session = Depends(get_db)):
    total_batches = db.query(func.count(MilkBatchRecord.id)).scalar() or 0
    if total_batches == 0:
        raise HTTPException(status_code=404, detail="Database is empty. Run seed_database.py first.")

    total_vol = db.query(func.sum(MilkBatchRecord.raw_milk_volume_liters)).scalar() or 0.0
    avg_temp = db.query(func.avg(MilkBatchRecord.reefer_chamber_temp_c)).scalar() or 0.0
    avg_ph = db.query(func.avg(MilkBatchRecord.milk_ph_level)).scalar() or 0.0
    total_saved_km = db.query(func.sum(MilkBatchRecord.optimal_route_saved_km)).scalar() or 0.0
    spoilage_count = db.query(func.count(MilkBatchRecord.id)).filter(
        MilkBatchRecord.cold_chain_spoilage_risk == 1
    ).scalar() or 0
    avg_demand = db.query(func.avg(MilkBatchRecord.next_day_demand_liters)).scalar() or 0.0

    hub_breakdown = (
        db.query(
            MilkBatchRecord.collection_center_bmc,
            func.count(MilkBatchRecord.id).label("batches"),
            func.sum(MilkBatchRecord.raw_milk_volume_liters).label("volume_liters"),
            func.avg(MilkBatchRecord.reefer_chamber_temp_c).label("avg_temp_c"),
            func.sum(MilkBatchRecord.cold_chain_spoilage_risk).label("risk_batches"),
        )
        .group_by(MilkBatchRecord.collection_center_bmc)
        .all()
    )

    return {
        "total_batches": total_batches,
        "total_volume_liters": round(float(total_vol), 1),
        "avg_reefer_temp_c": round(float(avg_temp), 2),
        "avg_milk_ph": round(float(avg_ph), 2),
        "total_route_saved_km": round(float(total_saved_km), 1),
        "spoilage_risk_count": spoilage_count,
        "spoilage_risk_pct": round((spoilage_count / total_batches) * 100.0, 2),
        "avg_next_day_demand_liters": round(float(avg_demand), 1),
        "hub_summary": [
            {
                "bmc_code": row[0],
                "batches": row[1],
                "volume_liters": round(float(row[2]), 1),
                "avg_temp_c": round(float(row[3]), 2),
                "risk_batches": int(row[4]),
            }
            for row in hub_breakdown
        ],
    }


@app.get("/api/hubs", response_model=List[BMCHubOut], tags=["Master Data"])
def list_bmc_hubs(db: Session = Depends(get_db)):
    return db.query(CollectionCenterBMC).all()


@app.get("/api/tankers", response_model=List[TankerOut], tags=["Master Data"])
def list_tankers(db: Session = Depends(get_db)):
    return db.query(RefrigeratedTanker).all()


@app.get("/api/batches", response_model=PaginatedBatchesResponse, tags=["Milk Batches"])
def list_milk_batches(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    bmc: Optional[str] = None,
    spoilage_risk: Optional[int] = Query(None, ge=0, le=1),
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(MilkBatchRecord)
    if bmc:
        query = query.filter(MilkBatchRecord.collection_center_bmc == bmc)
    if spoilage_risk is not None:
        query = query.filter(MilkBatchRecord.cold_chain_spoilage_risk == spoilage_risk)
    if search:
        like_term = f"%{search}%"
        query = query.filter(
            (MilkBatchRecord.batch_id.ilike(like_term))
            | (MilkBatchRecord.tanker_id.ilike(like_term))
            | (MilkBatchRecord.route_corridor.ilike(like_term))
        )

    total = query.count()
    items = (
        query.order_by(MilkBatchRecord.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.post("/api/batches", response_model=MilkBatchOut, status_code=201, tags=["Milk Batches"])
def create_milk_batch(payload: MilkBatchCreate, db: Session = Depends(get_db)):
    existing = db.query(MilkBatchRecord).filter(MilkBatchRecord.batch_id == payload.batch_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Batch {payload.batch_id} already exists.")
    record = MilkBatchRecord(**payload.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/api/batches/export/csv", tags=["Milk Batches"])
def export_batches_csv(db: Session = Depends(get_db)):
    records = db.query(MilkBatchRecord).order_by(MilkBatchRecord.id.asc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    headers = [
        "Batch_ID", "Timestamp", "Collection_Center_BMC", "Route_Corridor", "Tanker_ID",
        "Day_Of_Week", "Season_Index", "Raw_Milk_Volume_Liters", "Fat_Content_Pct",
        "SNF_Content_Pct", "Initial_Chilling_Temp_C", "Reefer_Chamber_Temp_C",
        "Ambient_External_Temp_C", "Transit_Distance_Km", "Transit_Duration_Hrs",
        "Reefer_Door_Open_Count", "Thermal_Excursion_Mins", "Milk_pH_Level",
        "MBRT_Hours", "Optimal_Route_Saved_Km", "Cold_Chain_Spoilage_Risk",
        "Next_Day_Demand_Liters"
    ]
    writer.writerow(headers)
    for r in records:
        writer.writerow([
            r.batch_id, r.timestamp.strftime("%Y-%m-%d %H:%M"), r.collection_center_bmc,
            r.route_corridor, r.tanker_id, r.day_of_week, r.season_index,
            r.raw_milk_volume_liters, r.fat_content_pct, r.snf_content_pct,
            r.initial_chilling_temp_c, r.reefer_chamber_temp_c, r.ambient_external_temp_c,
            r.transit_distance_km, r.transit_duration_hrs, r.reefer_door_open_count,
            r.thermal_excursion_mins, r.milk_ph_level, r.mbrt_hours,
            r.optimal_route_saved_km, r.cold_chain_spoilage_risk, r.next_day_demand_liters
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dairy_supply_chain_export.csv"},
    )


@app.post("/api/batches/upload/csv", tags=["Milk Batches"])
async def upload_batches_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = (await file.read()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    inserted = 0
    for row in reader:
        bid = row.get("Batch_ID")
        if not bid:
            continue
        if db.query(MilkBatchRecord).filter(MilkBatchRecord.batch_id == bid).first():
            continue
        rec = MilkBatchRecord(
            batch_id=bid,
            timestamp=datetime.strptime(row["Timestamp"], "%Y-%m-%d %H:%M"),
            collection_center_bmc=row["Collection_Center_BMC"],
            route_corridor=row["Route_Corridor"],
            tanker_id=row["Tanker_ID"],
            day_of_week=int(row["Day_Of_Week"]),
            season_index=int(row["Season_Index"]),
            raw_milk_volume_liters=float(row["Raw_Milk_Volume_Liters"]),
            fat_content_pct=float(row["Fat_Content_Pct"]),
            snf_content_pct=float(row["SNF_Content_Pct"]),
            initial_chilling_temp_c=float(row["Initial_Chilling_Temp_C"]),
            reefer_chamber_temp_c=float(row["Reefer_Chamber_Temp_C"]),
            ambient_external_temp_c=float(row["Ambient_External_Temp_C"]),
            transit_distance_km=float(row["Transit_Distance_Km"]),
            transit_duration_hrs=float(row["Transit_Duration_Hrs"]),
            reefer_door_open_count=int(row["Reefer_Door_Open_Count"]),
            thermal_excursion_mins=int(row["Thermal_Excursion_Mins"]),
            milk_ph_level=float(row["Milk_pH_Level"]),
            mbrt_hours=float(row["MBRT_Hours"]),
            optimal_route_saved_km=float(row["Optimal_Route_Saved_Km"]),
            cold_chain_spoilage_risk=int(row["Cold_Chain_Spoilage_Risk"]),
            next_day_demand_liters=float(row["Next_Day_Demand_Liters"]),
        )
        db.add(rec)
        inserted += 1
    db.commit()
    return {"filename": file.filename, "inserted_rows": inserted}


@app.post("/api/iot/telemetry", response_model=IoTTelemetryOut, status_code=201, tags=["IoT Cold-Chain"])
def ingest_iot_telemetry(payload: IoTTelemetryIngest, db: Session = Depends(get_db)):
    alert_triggered = False
    alert_msg = None
    if payload.reefer_temp_c > 4.5:
        alert_triggered = True
        alert_msg = f"NDDB Threshold Breach: Reefer chamber at {payload.reefer_temp_c}°C (> 4.5°C)."
    elif payload.door_status == "OPEN" and payload.ambient_temp_c > 32.0:
        alert_triggered = True
        alert_msg = f"Thermal Ingress Risk: Reefer door OPEN at {payload.ambient_temp_c}°C ambient."

    event = IoTTelemetryEvent(
        **payload.model_dump(),
        alert_triggered=alert_triggered,
        alert_message=alert_msg,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@app.get("/api/iot/telemetry/recent", response_model=List[IoTTelemetryOut], tags=["IoT Cold-Chain"])
def list_recent_telemetry(limit: int = Query(30, ge=1, le=200), db: Session = Depends(get_db)):
    return db.query(IoTTelemetryEvent).order_by(IoTTelemetryEvent.id.desc()).limit(limit).all()


@app.websocket("/ws/iot-stream")
async def websocket_iot_stream(websocket: WebSocket, inject_breach: bool = False):
    await websocket.accept()
    tankers = ["TKR-01", "TKR-03", "TKR-07", "TKR-12", "TKR-15"]
    try:
        while True:
            tkr = random.choice(tankers)
            if inject_breach and tkr == "TKR-03":
                temp = round(random.uniform(5.5, 6.4), 2)
                door = "OPEN"
            else:
                temp = round(random.uniform(2.8, 4.1), 2)
                door = "CLOSED"
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tanker_id": tkr,
                "reefer_temp_c": temp,
                "ambient_temp_c": round(random.uniform(27.0, 39.5), 1),
                "door_status": door,
                "alert": temp > 4.5,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass


@app.post("/api/ml/train", tags=["Machine Learning"])
def trigger_model_training(db: Session = Depends(get_db)):
    metrics = train_models_from_db(db)
    return {"status": "trained", "metrics": metrics}


@app.get("/api/ml/metrics", tags=["Machine Learning"])
def get_ml_metrics(db: Session = Depends(get_db)):
    bundle = load_or_train_bundle(db)
    latest_run = db.query(MLModelRun).order_by(MLModelRun.id.desc()).first()
    return {
        "metrics": bundle["metrics"],
        "latest_db_audit": {
            "id": latest_run.id if latest_run else None,
            "run_timestamp": latest_run.run_timestamp.isoformat() if latest_run else None,
            "training_rows": latest_run.training_rows if latest_run else None,
            "test_rows": latest_run.test_rows if latest_run else None,
        },
    }


@app.post("/api/ml/predict", response_model=PredictionResponse, tags=["Machine Learning"])
def predict_milk_batch(payload: PredictionRequest, db: Session = Depends(get_db)):
    return predict_batch_risk_and_demand(db, payload.model_dump())


@app.post("/api/routes/optimize", response_model=RouteOptimizeResponse, tags=["VRPTW Routing"])
def optimize_collection_routes(payload: RouteOptimizeRequest, db: Session = Depends(get_db)):
    stops_list = [s.model_dump() for s in payload.stops] if payload.stops else None
    return solve_vrptw_routes(db, tanker_capacity_liters=payload.tanker_capacity_liters, custom_stops=stops_list)
