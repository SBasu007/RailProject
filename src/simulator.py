"""Real-time Train Traffic Simulator with Digital Twin capabilities"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import json
import asyncio
from collections import defaultdict
import uuid

from src.models import (
    Train, Station, Section, TrainEvent, Conflict, 
    OptimizationSuggestion, SystemState, KPIMetrics,
    TrainStatus, EventType, ConflictType, Direction
)

class RailwaySimulator:
    def __init__(self, constants_file='constants.json'):
        with open(constants_file, 'r') as f:
            self.constants = json.load(f)
        
        self.stations = self._load_stations()
        self.sections = self._load_sections()
        self.trains = []
        self.events = []
        self.conflicts = []
        self.current_time = None
        self.simulation_horizon = timedelta(minutes=self.constants['scenario']['horizon_minutes'])
        
    def _load_stations(self) -> Dict[str, Station]:
        """Load stations from constants"""
        stations = {}
        for st in self.constants['stations']:
            platform_count = self.constants['platforms'].get(
                st['station_id'], {}).get('platform_count', 1
            )
            stations[st['station_id']] = Station(
                **st, platform_count=platform_count
            )
        return stations
    
    def _load_sections(self) -> Dict[str, Section]:
        """Load sections from constants"""
        sections = {}
        for sec in self.constants['sections']:
            sections[sec['section_id']] = Section(**sec)
        return sections
    
    def load_train_data(self, train_data: pd.DataFrame) -> List[Train]:
        """Load trains from CSV data"""
        trains = []
        for _, row in train_data.iterrows():
            train_type = row.get('train_type', 'passenger')
            defaults = self.constants['train_defaults'].get(train_type, {})
            
            train = Train(
                train_id=f"TRN_{row.get('train_number', uuid.uuid4().hex[:8])}",
                train_number=str(row.get('train_number', '')),
                name=row.get('name', f"Train {row.get('train_number', '')}"),
                train_type=train_type,
                priority=row.get('priority', defaults.get('priority', 5)),
                planned_speed_kmph=row.get('speed', defaults.get('planned_speed_kmph', 60)),
                max_allowed_speed_kmph=defaults.get('max_allowed_speed_kmph', 80),
                direction=row.get('direction', Direction.UP),
                wagons_count=row.get('wagons_count', defaults.get('wagons_count_default'))
            )
            trains.append(train)
        
        self.trains = trains
        return trains
    
    def detect_conflicts(
        self, 
        events: List[TrainEvent], 
        time_window: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Conflict]:
        """Detect conflicts in the schedule"""
        conflicts = []
        headway_sec = self.constants['operations']['headway_sec']
        
        # Group events by section and time
        section_events = defaultdict(list)
        platform_events = defaultdict(lambda: defaultdict(list))
        
        for event in events:
            if time_window:
                if not (time_window[0] <= event.scheduled_time <= time_window[1]):
                    continue
            
            # Track section occupancy
            train = next((t for t in self.trains if t.train_id == event.train_id), None)
            if train:
                # Find section for this movement
                for sec_id, section in self.sections.items():
                    if (event.event_type == EventType.DEPARTURE and 
                        event.station_id == section.from_station):
                        section_events[sec_id].append(event)
                    elif (event.event_type == EventType.ARRIVAL and 
                          event.station_id == section.to_station):
                        section_events[sec_id].append(event)
            
            # Track platform occupancy
            if event.platform is not None:
                platform_events[event.station_id][event.platform].append(event)
        
        # Check for section conflicts
        for section_id, events_list in section_events.items():
            events_list.sort(key=lambda x: x.scheduled_time)
            for i in range(len(events_list) - 1):
                time_diff = (events_list[i+1].scheduled_time - 
                           events_list[i].scheduled_time).total_seconds()
                
                if time_diff < headway_sec:
                    conflict = Conflict(
                        conflict_id=f"CNF_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.HEADWAY_VIOLATION,
                        severity="high" if time_diff < headway_sec/2 else "medium",
                        trains_involved=[events_list[i].train_id, events_list[i+1].train_id],
                        section_id=section_id,
                        time_window={
                            "start": events_list[i].scheduled_time,
                            "end": events_list[i+1].scheduled_time
                        },
                        description=f"Headway violation in section {section_id}: {time_diff:.0f}s < {headway_sec}s",
                        suggested_resolution=f"Delay train {events_list[i+1].train_id} by {headway_sec - time_diff:.0f} seconds"
                    )
                    conflicts.append(conflict)
        
        # Check for platform conflicts
        for station_id, platform_dict in platform_events.items():
            for platform_num, events_list in platform_dict.items():
                events_list.sort(key=lambda x: x.scheduled_time)
                for i in range(len(events_list) - 1):
                    # Check if platform is still occupied
                    if events_list[i].event_type in [EventType.ARRIVAL, EventType.DEPARTURE]:
                        min_dwell = self.constants['operations']['min_dwell_sec']
                        clear_time = events_list[i].scheduled_time + timedelta(seconds=min_dwell)
                        
                        if events_list[i+1].scheduled_time < clear_time:
                            conflict = Conflict(
                                conflict_id=f"CNF_{uuid.uuid4().hex[:8]}",
                                conflict_type=ConflictType.PLATFORM_CONFLICT,
                                severity="medium",
                                trains_involved=[events_list[i].train_id, events_list[i+1].train_id],
                                station_id=station_id,
                                time_window={
                                    "start": events_list[i].scheduled_time,
                                    "end": events_list[i+1].scheduled_time
                                },
                                description=f"Platform {platform_num} at {station_id} conflict",
                                suggested_resolution=f"Use alternate platform or delay train {events_list[i+1].train_id}"
                            )
                            conflicts.append(conflict)
        
        self.conflicts = conflicts
        return conflicts
    
    def propagate_delays(self, initial_delays: Dict[str, float]) -> Dict[str, float]:
        """Propagate delays through the network"""
        propagated_delays = initial_delays.copy()
        
        # Simple propagation: delays cascade to following trains
        for train_id, delay_minutes in initial_delays.items():
            # Find dependent trains (those following in same section)
            for event in self.events:
                if event.train_id == train_id:
                    # Find trains scheduled after this one in same section
                    for other_event in self.events:
                        if (other_event.train_id != train_id and
                            other_event.station_id == event.station_id and
                            other_event.scheduled_time > event.scheduled_time):
                            
                            time_gap = (other_event.scheduled_time - 
                                      event.scheduled_time).total_seconds() / 60
                            
                            if time_gap < delay_minutes:
                                # Propagate partial delay
                                cascade_delay = delay_minutes - time_gap
                                if other_event.train_id in propagated_delays:
                                    propagated_delays[other_event.train_id] = max(
                                        propagated_delays[other_event.train_id],
                                        cascade_delay
                                    )
                                else:
                                    propagated_delays[other_event.train_id] = cascade_delay
        
        return propagated_delays
    
    def calculate_metrics(self, delays: Optional[Dict[str, float]] = None) -> KPIMetrics:
        """Calculate KPI metrics for current state"""
        if delays is None:
            delays = {t.train_id: t.delay_minutes for t in self.trains}
        
        delay_values = list(delays.values())
        on_time_trains = sum(1 for d in delay_values if d <= 5)  # Within 5 min is on-time
        
        # Calculate section utilization
        total_capacity = len(self.sections) * self.simulation_horizon.total_seconds() / 60
        occupied_time = len(self.events) * self.constants['operations']['min_dwell_sec'] / 60
        
        metrics = KPIMetrics(
            timestamp=datetime.now(),
            section_throughput=len(self.trains) / (self.simulation_horizon.total_seconds() / 3600),
            average_delay_minutes=np.mean(delay_values) if delay_values else 0,
            max_delay_minutes=max(delay_values) if delay_values else 0,
            on_time_percentage=(on_time_trains / len(self.trains) * 100) if self.trains else 0,
            section_utilization=min(occupied_time / total_capacity, 1.0) if total_capacity > 0 else 0,
            conflict_count=len(self.conflicts),
            resolved_conflicts=0,
            train_count=len(self.trains),
            punctuality_score=max(0, 100 - np.mean(delay_values) * 10) if delay_values else 100
        )
        
        return metrics
    
    def simulate_step(self, current_time: datetime, time_step: timedelta = timedelta(minutes=1)):
        """Simulate one time step"""
        self.current_time = current_time
        
        # Update train positions
        for train in self.trains:
            if train.status == TrainStatus.RUNNING:
                # Calculate distance traveled
                distance_km = (train.current_speed_kmph * time_step.total_seconds()) / 3600
                
                # Update position (simplified)
                if train.current_position:
                    train.current_position['progress'] += distance_km
        
        # Check for events at current time
        current_events = [
            e for e in self.events 
            if abs((e.scheduled_time - current_time).total_seconds()) < 30
        ]
        
        # Process events
        for event in current_events:
            train = next((t for t in self.trains if t.train_id == event.train_id), None)
            if train:
                if event.event_type == EventType.ARRIVAL:
                    train.status = TrainStatus.ARRIVED
                elif event.event_type == EventType.DEPARTURE:
                    train.status = TrainStatus.RUNNING
        
        # Detect new conflicts
        time_window = (current_time, current_time + timedelta(minutes=30))
        self.detect_conflicts(self.events, time_window)
        
        return self.get_current_state()
    
    def get_current_state(self) -> SystemState:
        """Get current system state"""
        metrics = self.calculate_metrics()
        
        return SystemState(
            timestamp=self.current_time or datetime.now(),
            active_trains=self.trains,
            sections=list(self.sections.values()),
            stations=list(self.stations.values()),
            conflicts=self.conflicts,
            suggestions=[],  # Will be filled by optimizer
            kpis=metrics.dict()
        )
    
    def apply_suggestions(self, suggestions: List[OptimizationSuggestion]):
        """Apply optimization suggestions to the schedule"""
        for suggestion in suggestions:
            train = next((t for t in self.trains if t.train_id == suggestion.train_id), None)
            if not train:
                continue
            
            if suggestion.action_type == "delay":
                # Apply delay to train
                train.delay_minutes += suggestion.suggested_value
                # Update events
                for event in self.events:
                    if event.train_id == suggestion.train_id:
                        event.scheduled_time += timedelta(minutes=suggestion.suggested_value)
            
            elif suggestion.action_type == "speed_change":
                train.current_speed_kmph = suggestion.suggested_value
                train.planned_speed_kmph = suggestion.suggested_value
            
            elif suggestion.action_type == "platform_change":
                # Update platform assignments
                for event in self.events:
                    if (event.train_id == suggestion.train_id and 
                        event.station_id == suggestion.current_value['station']):
                        event.platform = suggestion.suggested_value
        
        # Re-detect conflicts after changes
        self.detect_conflicts(self.events)
        
        return self.get_current_state()
