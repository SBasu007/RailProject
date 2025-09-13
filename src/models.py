"""Core data models for the Rail Traffic Control System"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
import json

# Load constants
with open('constants.json', 'r') as f:
    CONSTANTS = json.load(f)

# Enums
class TrainType(str, Enum):
    PASSENGER = "passenger"
    FREIGHT = "freight"
    EXPRESS = "express"
    LOCAL = "local"
    MAINTENANCE = "maintenance"

class TrainStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    DELAYED = "delayed"
    ARRIVED = "arrived"
    CANCELLED = "cancelled"
    HALTED = "halted"

class Direction(str, Enum):
    UP = "UP"
    DOWN = "DN"

class EventType(str, Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"
    PASS = "PASS"

class ConflictType(str, Enum):
    SECTION_OVERLAP = "section_overlap"
    PLATFORM_CONFLICT = "platform_conflict"
    HEADWAY_VIOLATION = "headway_violation"
    CROSSING_CONFLICT = "crossing_conflict"

# Models
class Station(BaseModel):
    station_id: str
    name: str
    lat: float
    lon: float
    platform_count: int = 1
    platforms_occupied: List[str] = Field(default_factory=list)
    
class Section(BaseModel):
    section_id: str
    from_station: str
    to_station: str
    bidirectional: bool = True
    permitted_speed_kmph: float
    length_km: float
    current_occupancy: List[str] = Field(default_factory=list)
    gradient: Optional[float] = None
    signal_positions: List[float] = Field(default_factory=list)
    
class Train(BaseModel):
    train_id: str
    train_number: str
    name: str
    train_type: TrainType
    priority: int = Field(ge=1, le=10)
    status: TrainStatus = TrainStatus.SCHEDULED
    current_position: Optional[Dict[str, Any]] = None
    current_speed_kmph: float = 0
    planned_speed_kmph: float
    max_allowed_speed_kmph: float
    direction: Direction
    delay_minutes: float = 0
    wagons_count: Optional[int] = None
    
class TrainEvent(BaseModel):
    event_id: str
    train_id: str
    station_id: str
    event_type: EventType
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    platform: Optional[int] = None
    
class Conflict(BaseModel):
    conflict_id: str
    conflict_type: ConflictType
    severity: Literal["high", "medium", "low"]
    trains_involved: List[str]
    section_id: Optional[str] = None
    station_id: Optional[str] = None
    time_window: Dict[str, datetime]
    description: str
    suggested_resolution: Optional[str] = None
    
class OptimizationSuggestion(BaseModel):
    suggestion_id: str
    train_id: str
    action_type: Literal["delay", "speed_change", "route_change", "platform_change"]
    current_value: Any
    suggested_value: Any
    impact_minutes: float
    priority: int
    reason: str
    confidence_score: float = Field(ge=0, le=1)
    
class SystemState(BaseModel):
    timestamp: datetime
    active_trains: List[Train]
    sections: List[Section]
    stations: List[Station]
    conflicts: List[Conflict]
    suggestions: List[OptimizationSuggestion]
    kpis: Dict[str, float]
    
class SimulationRequest(BaseModel):
    scenario_name: str
    start_time: datetime
    end_time: datetime
    trains: List[Train]
    modifications: Optional[List[OptimizationSuggestion]] = None
    what_if_delays: Optional[Dict[str, float]] = None
    
class KPIMetrics(BaseModel):
    timestamp: datetime
    section_throughput: float
    average_delay_minutes: float
    max_delay_minutes: float
    on_time_percentage: float
    section_utilization: float
    conflict_count: int
    resolved_conflicts: int
    train_count: int
    punctuality_score: float