from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class BMCHubOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    bmc_code: str
    region_district: str
    latitude: float
    longitude: float
    base_distance_km: float
    chilling_capacity_liters: float
    hub_scale_factor: float
    active: bool


class TankerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    tanker_id: str
    vehicle_registration: str
    capacity_liters: float
    insulation_grade: str
    assigned_corridor: str
    status: str


class MilkBatchCreate(BaseModel):
    batch_id: str = Field(..., examples=["BATCH-12001"])
    timestamp: datetime
    collection_center_bmc: str = Field(..., examples=["BMC-Anand-North"])
    route_corridor: str = Field(..., examples=["CORR-A1-Express"])
    tanker_id: str = Field(..., examples=["TKR-01"])
    day_of_week: int = Field(..., ge=0, le=6)
    season_index: int = Field(..., ge=1, le=3)
    raw_milk_volume_liters: float = Field(..., gt=0)
    fat_content_pct: float = Field(..., ge=2.0, le=10.0)
    snf_content_pct: float = Field(..., ge=6.0, le=12.0)
    initial_chilling_temp_c: float = Field(..., ge=0.0, le=15.0)
    reefer_chamber_temp_c: float = Field(..., ge=0.0, le=25.0)
    ambient_external_temp_c: float = Field(..., ge=-5.0, le=55.0)
    transit_distance_km: float = Field(..., gt=0)
    transit_duration_hrs: float = Field(..., gt=0)
    reefer_door_open_count: int = Field(0, ge=0)
    thermal_excursion_mins: int = Field(0, ge=0)
    milk_ph_level: float = Field(..., ge=5.0, le=7.5)
    mbrt_hours: float = Field(..., ge=0.1, le=10.0)
    optimal_route_saved_km: float = Field(0.0, ge=0.0)
    cold_chain_spoilage_risk: int = Field(..., ge=0, le=1)
    next_day_demand_liters: float = Field(..., gt=0)


class MilkBatchOut(MilkBatchCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int


class PaginatedBatchesResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[MilkBatchOut]


class IoTTelemetryIngest(BaseModel):
    tanker_id: str = Field(..., examples=["TKR-03"])
    batch_id: Optional[str] = Field(None, examples=["BATCH-11499"])
    latitude: float = Field(..., examples=[22.5645])
    longitude: float = Field(..., examples=[72.9289])
    reefer_temp_c: float = Field(..., examples=[5.4])
    ambient_temp_c: float = Field(..., examples=[36.2])
    compressor_rpm: int = Field(2400, ge=0, le=5000)
    door_status: str = Field("CLOSED", pattern="^(OPEN|CLOSED)$")
    battery_voltage: float = Field(24.1, ge=10.0, le=30.0)


class IoTTelemetryOut(IoTTelemetryIngest):
    model_config = ConfigDict(from_attributes=True)
    id: int
    recorded_at: datetime
    alert_triggered: bool
    alert_message: Optional[str] = None


class PredictionRequest(BaseModel):
    collection_center_bmc: str = Field("BMC-Banaskantha-West")
    day_of_week: int = Field(4, ge=0, le=6)
    season_index: int = Field(2, ge=1, le=3)
    raw_milk_volume_liters: float = Field(6200.0, gt=0)
    fat_content_pct: float = Field(4.4, ge=2.0, le=10.0)
    snf_content_pct: float = Field(8.65, ge=6.0, le=12.0)
    initial_chilling_temp_c: float = Field(3.2, ge=0.0, le=15.0)
    reefer_chamber_temp_c: float = Field(5.8, ge=0.0, le=25.0)
    ambient_external_temp_c: float = Field(38.0, ge=-5.0, le=55.0)
    transit_distance_km: float = Field(115.0, gt=0)
    transit_duration_hrs: float = Field(3.6, gt=0)
    reefer_door_open_count: int = Field(3, ge=0)
    thermal_excursion_mins: int = Field(65, ge=0)
    milk_ph_level: float = Field(6.48, ge=5.0, le=7.5)
    mbrt_hours: float = Field(2.4, ge=0.1, le=10.0)


class PredictionResponse(BaseModel):
    spoilage_probability_rf: float
    spoilage_probability_logistic: float
    spoilage_risk_label: str
    recommended_action: str
    predicted_next_day_demand_liters: float
    feature_contributions: Dict[str, float]


class RouteStopInput(BaseModel):
    bmc_code: str
    demand_liters: float = Field(..., gt=0)
    urgency_temp_c: float = Field(3.5, ge=0.0, le=15.0)


class RouteOptimizeRequest(BaseModel):
    tanker_capacity_liters: float = Field(15000.0, gt=1000)
    stops: Optional[List[RouteStopInput]] = None


class RouteOptimizeResponse(BaseModel):
    plan_code: str
    depot_name: str
    total_stops: int
    total_volume_liters: float
    baseline_distance_km: float
    optimized_distance_km: float
    distance_saved_km: float
    distance_reduction_pct: float
    fuel_saved_liters: float
    co2_reduced_kg: float
    assigned_routes: List[Dict[str, Any]]
