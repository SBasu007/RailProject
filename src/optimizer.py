"""AI-Powered Optimization Engine for Railway Traffic Control"""
import numpy as np
from ortools.sat.python import cp_model
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid
from collections import defaultdict

from src.models import (
    Train, Section, Station, Conflict, OptimizationSuggestion,
    ConflictType, TrainEvent, EventType
)
from src.simulator import RailwaySimulator

class RailwayOptimizer:
    def __init__(self, simulator: RailwaySimulator):
        self.simulator = simulator
        self.constants = simulator.constants
        self.model = None
        self.solver = None
        
    def optimize_schedule(
        self, 
        conflicts: List[Conflict],
        horizon_minutes: int = 60
    ) -> List[OptimizationSuggestion]:
        """Main optimization function using CP-SAT solver"""
        
        if not conflicts:
            return []
        
        suggestions = []
        
        # Use heuristic approach for quick resolution
        heuristic_suggestions = self._heuristic_resolution(conflicts)
        
        # Use CP-SAT for complex scenarios
        if len(conflicts) > 3:
            cp_suggestions = self._cp_sat_optimization(conflicts, horizon_minutes)
            if cp_suggestions:
                suggestions = cp_suggestions
            else:
                suggestions = heuristic_suggestions
        else:
            suggestions = heuristic_suggestions
        
        return suggestions
    
    def _heuristic_resolution(self, conflicts: List[Conflict]) -> List[OptimizationSuggestion]:
        """Quick heuristic-based conflict resolution"""
        suggestions = []
        
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.HEADWAY_VIOLATION:
                # Delay lower priority train
                trains = [self._get_train(tid) for tid in conflict.trains_involved]
                trains_with_priority = [(t, t.priority if t else 999) for t in trains]
                trains_with_priority.sort(key=lambda x: x[1])
                
                if len(trains_with_priority) >= 2:
                    lower_priority_train = trains_with_priority[-1][0]
                    if lower_priority_train:
                        delay_needed = self.constants['operations']['headway_sec'] / 60  # Convert to minutes
                        
                        suggestion = OptimizationSuggestion(
                            suggestion_id=f"SUG_{uuid.uuid4().hex[:8]}",
                            train_id=lower_priority_train.train_id,
                            action_type="delay",
                            current_value=lower_priority_train.delay_minutes,
                            suggested_value=delay_needed,
                            impact_minutes=delay_needed,
                            priority=1,
                            reason=f"Resolve headway conflict in section {conflict.section_id}",
                            confidence_score=0.85
                        )
                        suggestions.append(suggestion)
            
            elif conflict.conflict_type == ConflictType.PLATFORM_CONFLICT:
                # Try platform change first, then delay
                station = self.simulator.stations.get(conflict.station_id)
                if station and station.platform_count > 1:
                    # Suggest platform change
                    train_id = conflict.trains_involved[1]  # Second train
                    suggestion = OptimizationSuggestion(
                        suggestion_id=f"SUG_{uuid.uuid4().hex[:8]}",
                        train_id=train_id,
                        action_type="platform_change",
                        current_value={'station': conflict.station_id, 'platform': 1},
                        suggested_value=2,  # Alternative platform
                        impact_minutes=0,
                        priority=2,
                        reason=f"Use alternative platform at {station.name}",
                        confidence_score=0.90
                    )
                    suggestions.append(suggestion)
                else:
                    # Delay second train
                    train_id = conflict.trains_involved[1]
                    delay_needed = self.constants['operations']['min_dwell_sec'] / 60
                    
                    suggestion = OptimizationSuggestion(
                        suggestion_id=f"SUG_{uuid.uuid4().hex[:8]}",
                        train_id=train_id,
                        action_type="delay",
                        current_value=0,
                        suggested_value=delay_needed,
                        impact_minutes=delay_needed,
                        priority=1,
                        reason=f"Platform conflict at {conflict.station_id}",
                        confidence_score=0.80
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _cp_sat_optimization(
        self, 
        conflicts: List[Conflict],
        horizon_minutes: int
    ) -> List[OptimizationSuggestion]:
        """Advanced optimization using Google OR-Tools CP-SAT solver"""
        
        model = cp_model.CpModel()
        
        # Decision variables
        train_delays = {}  # Delay for each train (in minutes)
        train_speeds = {}  # Speed adjustment for each train
        
        # Get all involved trains
        involved_trains = set()
        for conflict in conflicts:
            involved_trains.update(conflict.trains_involved)
        
        # Create variables for each train
        max_delay = self.constants['optimizer']['max_offset_minutes']
        
        for train_id in involved_trains:
            train = self._get_train(train_id)
            if train:
                # Delay variable (0 to max_delay minutes)
                train_delays[train_id] = model.NewIntVar(0, max_delay * 60, f'delay_{train_id}')
                
                # Speed variable (50% to 100% of planned speed)
                min_speed = int(train.planned_speed_kmph * 0.5)
                max_speed = int(train.max_allowed_speed_kmph)
                train_speeds[train_id] = model.NewIntVar(min_speed, max_speed, f'speed_{train_id}')
        
        # Add constraints for each conflict
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.HEADWAY_VIOLATION:
                if len(conflict.trains_involved) >= 2:
                    train1_id = conflict.trains_involved[0]
                    train2_id = conflict.trains_involved[1]
                    
                    if train1_id in train_delays and train2_id in train_delays:
                        # Ensure minimum headway between trains
                        headway_sec = self.constants['operations']['headway_sec']
                        model.Add(train_delays[train2_id] >= train_delays[train1_id] + headway_sec)
        
        # Objective: Minimize weighted total delay
        weights = self.constants['optimizer']['priority_weights']
        objective_terms = []
        
        for train_id in train_delays:
            train = self._get_train(train_id)
            if train:
                weight = weights.get(train.train_type.value, 1.0)
                # Multiply by 10 to work with integer arithmetic
                weighted_delay = model.NewIntVar(0, int(max_delay * 60 * weight * 10), f'weighted_{train_id}')
                model.Add(weighted_delay == train_delays[train_id] * int(weight * 10))
                objective_terms.append(weighted_delay)
        
        model.Minimize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # Time limit
        status = solver.Solve(model)
        
        suggestions = []
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for train_id in train_delays:
                delay_seconds = solver.Value(train_delays[train_id])
                if delay_seconds > 0:
                    delay_minutes = delay_seconds / 60.0
                    
                    suggestion = OptimizationSuggestion(
                        suggestion_id=f"SUG_{uuid.uuid4().hex[:8]}",
                        train_id=train_id,
                        action_type="delay",
                        current_value=0,
                        suggested_value=delay_minutes,
                        impact_minutes=delay_minutes,
                        priority=1,
                        reason="CP-SAT optimized schedule adjustment",
                        confidence_score=0.95
                    )
                    suggestions.append(suggestion)
                
                # Check for speed changes
                if train_id in train_speeds:
                    suggested_speed = solver.Value(train_speeds[train_id])
                    train = self._get_train(train_id)
                    if train and abs(suggested_speed - train.planned_speed_kmph) > 5:
                        suggestion = OptimizationSuggestion(
                            suggestion_id=f"SUG_{uuid.uuid4().hex[:8]}",
                            train_id=train_id,
                            action_type="speed_change",
                            current_value=train.planned_speed_kmph,
                            suggested_value=suggested_speed,
                            impact_minutes=0,
                            priority=2,
                            reason="Speed optimization for conflict resolution",
                            confidence_score=0.85
                        )
                        suggestions.append(suggestion)
        
        return suggestions
    
    def _get_train(self, train_id: str) -> Optional[Train]:
        """Helper to get train by ID"""
        return next((t for t in self.simulator.trains if t.train_id == train_id), None)
    
    def evaluate_suggestions(
        self, 
        suggestions: List[OptimizationSuggestion]
    ) -> Dict[str, float]:
        """Evaluate the impact of suggestions"""
        
        # Create a copy of current state
        original_delays = {t.train_id: t.delay_minutes for t in self.simulator.trains}
        
        # Apply suggestions temporarily
        self.simulator.apply_suggestions(suggestions)
        
        # Calculate new metrics
        new_metrics = self.simulator.calculate_metrics()
        
        # Restore original state
        for train in self.simulator.trains:
            train.delay_minutes = original_delays.get(train.train_id, 0)
        
        return {
            "average_delay_reduction": new_metrics.average_delay_minutes,
            "conflicts_resolved": len(self.simulator.conflicts),
            "throughput_improvement": new_metrics.section_throughput,
            "punctuality_score": new_metrics.punctuality_score
        }
    
    def generate_what_if_scenarios(
        self,
        base_suggestions: List[OptimizationSuggestion],
        num_scenarios: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate what-if scenarios for decision support"""
        
        scenarios = []
        
        # Scenario 1: Apply all suggestions
        all_applied = {
            "name": "Apply All Optimizations",
            "suggestions": base_suggestions,
            "impact": self.evaluate_suggestions(base_suggestions)
        }
        scenarios.append(all_applied)
        
        # Scenario 2: Apply only high-priority suggestions
        high_priority = [s for s in base_suggestions if s.priority == 1]
        if high_priority:
            high_priority_scenario = {
                "name": "High Priority Only",
                "suggestions": high_priority,
                "impact": self.evaluate_suggestions(high_priority)
            }
            scenarios.append(high_priority_scenario)
        
        # Scenario 3: Minimal intervention (only critical conflicts)
        critical = [s for s in base_suggestions if s.confidence_score > 0.9]
        if critical:
            minimal_scenario = {
                "name": "Minimal Intervention",
                "suggestions": critical,
                "impact": self.evaluate_suggestions(critical)
            }
            scenarios.append(minimal_scenario)
        
        return scenarios
