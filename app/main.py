"""FastAPI Backend for Railway Traffic Control System"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
import uuid
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import RailwaySimulator
from src.optimizer import RailwayOptimizer
from src.models import (
    Train, Station, Section, Conflict, OptimizationSuggestion,
    SystemState, SimulationRequest, KPIMetrics, TrainEvent,
    TrainType, TrainStatus, Direction, EventType
)

app = FastAPI(title="Railway Traffic Control System", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
simulator = RailwaySimulator()
optimizer = RailwayOptimizer(simulator)
connected_clients: List[WebSocket] = []
simulation_task = None
simulation_running = False

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Initialize with sample data
def initialize_sample_data():
    """Load sample train data for demonstration"""
    sample_trains = []
    base_time = datetime.now().replace(second=0, microsecond=0)
    
    # Create sample trains
    train_configs = [
        {"number": "12345", "name": "Rajdhani Express", "type": "express", "priority": 1, "speed": 110},
        {"number": "22456", "name": "Shatabdi Express", "type": "express", "priority": 1, "speed": 100},
        {"number": "53421", "name": "Local Passenger", "type": "local", "priority": 3, "speed": 60},
        {"number": "63522", "name": "MEMU Local", "type": "local", "priority": 3, "speed": 55},
        {"number": "70012", "name": "Freight Train", "type": "freight", "priority": 5, "speed": 45},
    ]
    
    for i, config in enumerate(train_configs):
        train = Train(
            train_id=f"TRN_{config['number']}",
            train_number=config["number"],
            name=config["name"],
            train_type=TrainType(config["type"]),
            priority=config["priority"],
            planned_speed_kmph=config["speed"],
            max_allowed_speed_kmph=config["speed"] * 1.2,
            direction=Direction.UP if i % 2 == 0 else Direction.DOWN,
            status=TrainStatus.SCHEDULED,
            delay_minutes=0
        )
        sample_trains.append(train)
    
    simulator.trains = sample_trains
    
    # Create sample events
    sample_events = []
    stations = list(simulator.stations.keys())
    
    for i, train in enumerate(sample_trains):
        for j, station_id in enumerate(stations):
            event_time = base_time + timedelta(minutes=15*i + 20*j)
            
            # Arrival event
            if j > 0:
                arrival_event = TrainEvent(
                    event_id=f"EVT_{uuid.uuid4().hex[:8]}",
                    train_id=train.train_id,
                    station_id=station_id,
                    event_type=EventType.ARRIVAL,
                    scheduled_time=event_time,
                    platform=1
                )
                sample_events.append(arrival_event)
            
            # Departure event
            if j < len(stations) - 1:
                departure_event = TrainEvent(
                    event_id=f"EVT_{uuid.uuid4().hex[:8]}",
                    train_id=train.train_id,
                    station_id=station_id,
                    event_type=EventType.DEPARTURE,
                    scheduled_time=event_time + timedelta(minutes=2),
                    platform=1
                )
                sample_events.append(departure_event)
    
    simulator.events = sample_events
    
    # Detect initial conflicts
    simulator.detect_conflicts(sample_events)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    initialize_sample_data()
    print("Railway Traffic Control System initialized")

@app.get("/")
async def root():
    """Serve the main dashboard"""
    return FileResponse("app/static/index.html")

@app.get("/api/system/state")
async def get_system_state():
    """Get current system state"""
    state = simulator.get_current_state()
    return state.dict()

@app.get("/api/trains")
async def get_trains():
    """Get all trains"""
    return [train.dict() for train in simulator.trains]

@app.get("/api/trains/{train_id}")
async def get_train(train_id: str):
    """Get specific train details"""
    train = next((t for t in simulator.trains if t.train_id == train_id), None)
    if not train:
        raise HTTPException(status_code=404, detail="Train not found")
    return train.dict()

@app.post("/api/trains/{train_id}/delay")
async def apply_train_delay(train_id: str, delay_minutes: float):
    """Apply delay to a specific train"""
    train = next((t for t in simulator.trains if t.train_id == train_id), None)
    if not train:
        raise HTTPException(status_code=404, detail="Train not found")
    
    train.delay_minutes = delay_minutes
    
    # Propagate delays
    delays = {train_id: delay_minutes}
    propagated = simulator.propagate_delays(delays)
    
    # Broadcast update
    await manager.broadcast({
        "type": "delay_update",
        "data": propagated
    })
    
    return {"status": "success", "propagated_delays": propagated}

@app.get("/api/sections")
async def get_sections():
    """Get all sections"""
    return [section.dict() for section in simulator.sections.values()]

@app.get("/api/stations")
async def get_stations():
    """Get all stations"""
    return [station.dict() for station in simulator.stations.values()]

@app.get("/api/conflicts")
async def get_conflicts():
    """Get current conflicts"""
    return [conflict.dict() for conflict in simulator.conflicts]

@app.post("/api/conflicts/detect")
async def detect_conflicts():
    """Manually trigger conflict detection"""
    conflicts = simulator.detect_conflicts(simulator.events)
    return [c.dict() for c in conflicts]

@app.get("/api/suggestions")
async def get_optimization_suggestions():
    """Get optimization suggestions for current conflicts"""
    suggestions = optimizer.optimize_schedule(simulator.conflicts)
    return [s.dict() for s in suggestions]

@app.post("/api/suggestions/apply")
async def apply_suggestions(suggestion_ids: List[str]):
    """Apply selected optimization suggestions"""
    suggestions = optimizer.optimize_schedule(simulator.conflicts)
    selected = [s for s in suggestions if s.suggestion_id in suggestion_ids]
    
    if not selected:
        raise HTTPException(status_code=404, detail="No valid suggestions found")
    
    simulator.apply_suggestions(selected)
    
    # Broadcast update
    await manager.broadcast({
        "type": "suggestions_applied",
        "data": [s.dict() for s in selected]
    })
    
    return {"status": "success", "applied": len(selected)}

@app.get("/api/scenarios")
async def get_what_if_scenarios():
    """Generate what-if scenarios"""
    suggestions = optimizer.optimize_schedule(simulator.conflicts)
    scenarios = optimizer.generate_what_if_scenarios(suggestions)
    return scenarios

@app.get("/api/metrics")
async def get_metrics():
    """Get current KPI metrics"""
    metrics = simulator.calculate_metrics()
    return metrics.dict()

@app.get("/api/metrics/history")
async def get_metrics_history(hours: int = 1):
    """Get metrics history (mock data for demo)"""
    # Generate mock historical data
    history = []
    base_time = datetime.now()
    
    for i in range(hours * 4):  # Every 15 minutes
        timestamp = base_time - timedelta(minutes=15 * i)
        metrics = KPIMetrics(
            timestamp=timestamp,
            section_throughput=10 + (i % 3),
            average_delay_minutes=5 + (i % 5),
            max_delay_minutes=15 + (i % 10),
            on_time_percentage=85 - (i % 10),
            section_utilization=0.7 + (i % 3) * 0.05,
            conflict_count=2 + (i % 4),
            resolved_conflicts=1 + (i % 3),
            train_count=5,
            punctuality_score=90 - (i % 8)
        )
        history.append(metrics.dict())
    
    return history

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        state = simulator.get_current_state()
        await websocket.send_json({
            "type": "initial_state",
            "data": state.dict()
        })
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_simulation":
                asyncio.create_task(run_simulation())
            elif message["type"] == "stop_simulation":
                global simulation_running
                simulation_running = False
            elif message["type"] == "get_state":
                state = simulator.get_current_state()
                await websocket.send_json({
                    "type": "state_update",
                    "data": state.dict()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def run_simulation():
    """Run real-time simulation"""
    global simulation_running
    simulation_running = True
    
    start_time = datetime.now()
    time_step = timedelta(seconds=1)  # Simulation step
    broadcast_interval = 2  # Broadcast every 2 seconds
    last_broadcast = datetime.now()
    
    while simulation_running:
        current_time = datetime.now()
        
        # Step the simulation
        state = simulator.simulate_step(current_time, time_step)
        
        # Check for new conflicts
        if simulator.conflicts:
            # Get optimization suggestions
            suggestions = optimizer.optimize_schedule(simulator.conflicts)
            state.suggestions = suggestions
        
        # Broadcast updates at intervals
        if (current_time - last_broadcast).total_seconds() >= broadcast_interval:
            await manager.broadcast({
                "type": "simulation_update",
                "data": {
                    "timestamp": current_time.isoformat(),
                    "state": state.dict(),
                    "metrics": simulator.calculate_metrics().dict()
                }
            })
            last_broadcast = current_time
        
        await asyncio.sleep(1)

# Simulation control endpoints
@app.post("/api/simulation/start")
async def start_simulation(background_tasks: BackgroundTasks):
    """Start the simulation"""
    global simulation_task, simulation_running
    
    if not simulation_running:
        simulation_running = True
        background_tasks.add_task(run_simulation)
        return {"status": "started"}
    return {"status": "already_running"}

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop the simulation"""
    global simulation_running
    simulation_running = False
    return {"status": "stopped"}

@app.post("/api/simulation/reset")
async def reset_simulation():
    """Reset the simulation to initial state"""
    global simulation_running
    simulation_running = False
    
    # Reinitialize
    initialize_sample_data()
    
    # Broadcast reset
    await manager.broadcast({
        "type": "simulation_reset",
        "data": simulator.get_current_state().dict()
    })
    
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)