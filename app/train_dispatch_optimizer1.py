"""
Train Dispatch Optimization System using Google OR-Tools CP-SAT Solver
Optimizes train schedules considering multiple constraints and objectives
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import random

class TrainDispatchOptimizer:
    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.trains = []
        self.stations = set()
        self.maintenance_windows = []
        self.blocks = {}
        self.variables = {}
        self.solution = {}
        
        # Constants
        self.DISTANCE_BETWEEN_STATIONS = 2000  # meters
        self.BLOCK_SIZE = 400  # meters
        self.BLOCKS_PER_SEGMENT = 5
        self.MIN_DWELL_TIME = 1  # minutes
        self.MAX_EARLINESS = 5  # minutes
        self.MAX_LATENESS = 5  # minutes
        self.MIN_HEADWAY = 2  # minutes
        self.MIN_TRANSFER_TIME = 3  # minutes
        self.MAX_SPEED = 60  # km/h
        self.TIME_HORIZON = 200  # minutes (max time for scheduling)
        
        # Priority weights for delays
        self.PRIORITY_WEIGHTS = {
            'express': 10,
            'local': 5,
            'freight': 1
        }

    def load_data(self, filename='trains.json'):
        """Load train data from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.trains = json.load(f)
            
            # Extract all unique stations
            for train in self.trains:
                self.stations.update(set(train['route']))  # Convert to set first
            
            # Create blocks for each segment
            self._create_blocks()
            
            print(f"Loaded {len(self.trains)} trains")
            print(f"Stations: {sorted(self.stations)}")
            print(f"Created {len(self.blocks)} blocks")
            
        except FileNotFoundError:
            print(f"File {filename} not found. Creating sample data...")
            self._create_sample_data()
            self.load_data(filename)

    def load_maintenance_data(self, filename='maintenance.json'):
        """Load maintenance window data from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.maintenance_windows = json.load(f)
            
            print(f"Loaded {len(self.maintenance_windows)} maintenance windows")
            
        except FileNotFoundError:
            print(f"File {filename} not found. Creating sample maintenance data...")
            self._create_sample_maintenance_data()
            self.load_maintenance_data(filename)
    
    def _create_sample_data(self):
        """Create sample trains.json file"""
        sample_data = [
            {"train": "E101", "route": ["S1", "S2", "S3", "S4"], "planned": [0, 12, 25, 40], "type": "express"},
            {"train": "L201", "route": ["S1", "S2", "S3", "S4"], "planned": [5, 18, 32, 48], "type": "local"},
            {"train": "E102", "route": ["S1", "S2", "S3"], "planned": [10, 22, 35], "type": "express"},
            {"train": "F301", "route": ["S2", "S3", "S4"], "planned": [8, 20, 35], "type": "freight"}
        ]
        
        with open('trains.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print("Created sample trains.json file")
    
    def _create_sample_maintenance_data(self):
        """Create sample maintenance.json file"""
        sample_data = [
            {"block": "S1-S2-B3", "start_time": 10, "end_time": 20},
            {"block": "S2-S3-B2", "start_time": 25, "end_time": 35},
            {"block": "S3-S4-B1", "start_time": 40, "end_time": 50}
        ]
        
        with open('maintenance.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print("Created sample maintenance.json file")
    
    def _create_blocks(self):
        """Create block segments between stations"""
        station_list = sorted(list(self.stations))
        for i in range(len(station_list) - 1):
            segment_name = f"{station_list[i]}-{station_list[i+1]}"
            self.blocks[segment_name] = []
            for b in range(self.BLOCKS_PER_SEGMENT):
                block_id = f"{segment_name}_B{b+1}"
                self.blocks[segment_name].append(block_id)
    
    def build_model(self):
        """Build the CP-SAT model with all constraints"""
        print("\nBuilding optimization model...")
        
        # Create decision variables
        self._create_variables()
        
        # Add constraints
        self._add_temporal_constraints()
        self._add_spatial_constraints()
        self._add_operational_constraints()
        self._add_maintenance_constraints()
        
        # Define objective function
        self._define_objective()
        
        print("Model built successfully")
    
    def _create_variables(self):
        """Create decision variables for the model"""
        for train in self.trains:
            train_id = train['train']
            self.variables[train_id] = {
                'arrival': {},
                'departure': {},
                'block_entry': {},
                'block_exit': {},
                'block_occupied': {},
                'delay': {}
            }
            
            # Variables for each station
            for i, station in enumerate(train['route']):
                # Arrival time at station
                self.variables[train_id]['arrival'][station] = self.model.NewIntVar(
                    0, self.TIME_HORIZON, f'{train_id}_arr_{station}'
                )
                
                # Departure time from station
                self.variables[train_id]['departure'][station] = self.model.NewIntVar(
                    0, self.TIME_HORIZON, f'{train_id}_dep_{station}'
                )
                
                # Delay at station (for objective)
                self.variables[train_id]['delay'][station] = self.model.NewIntVar(
                    -self.MAX_EARLINESS, self.MAX_LATENESS * 2, f'{train_id}_delay_{station}'
                )
                
                # Create block occupancy variables for segments
                if i < len(train['route']) - 1:
                    next_station = train['route'][i + 1]
                    segment = f"{station}-{next_station}"
                    if segment in self.blocks:
                        for block in self.blocks[segment]:
                            # Entry time to block
                            self.variables[train_id]['block_entry'][block] = self.model.NewIntVar(
                                0, self.TIME_HORIZON, f'{train_id}_entry_{block}'
                            )
                            # Exit time from block
                            self.variables[train_id]['block_exit'][block] = self.model.NewIntVar(
                                0, self.TIME_HORIZON, f'{train_id}_exit_{block}'
                            )
                            # Boolean: is block occupied by this train
                            self.variables[train_id]['block_occupied'][block] = self.model.NewBoolVar(
                                f'{train_id}_occ_{block}'
                            )
    
    def _add_temporal_constraints(self):
        """Add temporal/scheduling constraints"""
        print("Adding temporal constraints...")
        
        for train in self.trains:
            train_id = train['train']
            vars = self.variables[train_id]
            
            for i, station in enumerate(train['route']):
                planned_time = train['planned'][i]
                
                # 1. Planned arrival window constraints
                self.model.Add(
                    vars['arrival'][station] >= planned_time - self.MAX_EARLINESS
                )
                self.model.Add(
                    vars['arrival'][station] <= planned_time + self.MAX_LATENESS
                )
                
                # 2. Minimum dwell time at station
                self.model.Add(
                    vars['departure'][station] >= vars['arrival'][station] + self.MIN_DWELL_TIME
                )
                
                # 3. Delay calculation for objective
                self.model.Add(
                    vars['delay'][station] == vars['arrival'][station] - planned_time
                )
                
                # 4. Travel time constraints between stations
                if i > 0:
                    prev_station = train['route'][i - 1]
                    min_travel_time = train['planned'][i] - train['planned'][i - 1]
                    # Ensure minimum travel time
                    self.model.Add(
                        vars['arrival'][station] >= vars['departure'][prev_station] + max(min_travel_time, 5)
                    )
                
                # 5. Block sequencing constraints
                if i < len(train['route']) - 1:
                    next_station = train['route'][i + 1]
                    segment = f"{station}-{next_station}"
                    if segment in self.blocks:
                        blocks = self.blocks[segment]
                        
                        # First block entry after station departure
                        self.model.Add(
                            vars['block_entry'][blocks[0]] >= vars['departure'][station]
                        )
                        
                        # Last block exit before next station arrival
                        self.model.Add(
                            vars['block_exit'][blocks[-1]] <= vars['arrival'][next_station]
                        )
                        
                        # Sequential block traversal
                        for j in range(len(blocks) - 1):
                            self.model.Add(
                                vars['block_entry'][blocks[j + 1]] >= vars['block_exit'][blocks[j]]
                            )
                        
                        # Minimum block traversal time (based on speed limit)
                        block_traverse_time = int((self.BLOCK_SIZE / 1000) / (self.MAX_SPEED / 60))  # minutes
                        for block in blocks:
                            self.model.Add(
                                vars['block_exit'][block] >= vars['block_entry'][block] + block_traverse_time
                            )
    
    def _add_spatial_constraints(self):
        """Add spatial/capacity constraints"""
        print("Adding spatial constraints...")
        
        # 1. Block occupancy constraints - no two trains in same block simultaneously
        for segment, blocks in self.blocks.items():
            for block in blocks:
                # Collect all trains that might use this block
                trains_using_block = []
                for train in self.trains:
                    train_id = train['train']
                    if block in self.variables[train_id]['block_entry']:
                        trains_using_block.append(train_id)
                
                # For each pair of trains, ensure no overlap in block occupancy
                for i in range(len(trains_using_block)):
                    for j in range(i + 1, len(trains_using_block)):
                        train1 = trains_using_block[i]
                        train2 = trains_using_block[j]
                        
                        # Create size variable for intervals (must be positive constant)
                        size1 = self.model.NewIntVar(1, 100, f'{train1}_{block}_size')
                        size2 = self.model.NewIntVar(1, 100, f'{train2}_{block}_size')
                        
                        # Set size values
                        self.model.Add(
                            size1 == self.variables[train1]['block_exit'][block] - self.variables[train1]['block_entry'][block]
                        )
                        self.model.Add(
                            size2 == self.variables[train2]['block_exit'][block] - self.variables[train2]['block_entry'][block]
                        )
                        
                        # Create interval variables for block occupancy
                        interval1 = self.model.NewOptionalIntervalVar(
                            self.variables[train1]['block_entry'][block],
                            size1,
                            self.variables[train1]['block_exit'][block],
                            self.variables[train1]['block_occupied'][block],
                            f'{train1}_{block}_interval'
                        )
                        
                        interval2 = self.model.NewOptionalIntervalVar(
                            self.variables[train2]['block_entry'][block],
                            size2,
                            self.variables[train2]['block_exit'][block],
                            self.variables[train2]['block_occupied'][block],
                            f'{train2}_{block}_interval'
                        )
                        
                        # No overlap constraint
                        self.model.AddNoOverlap([interval1, interval2])
        
        # 2. Platform capacity - one train per station platform at a time
        for station in self.stations:
            trains_at_station = []
            for train in self.trains:
                train_id = train['train']
                if station in train['route']:
                    trains_at_station.append(train_id)
            
            # Create intervals for platform occupancy
            intervals = []
            for train_id in trains_at_station:
                if station in self.variables[train_id]['arrival']:
                    # Create size variable for intervals (must be positive constant)
                    size = self.model.NewIntVar(1, 100, f'{train_id}_{station}_size')
                    
                    # Set size values
                    self.model.Add(
                        size == self.variables[train_id]['departure'][station] - self.variables[train_id]['arrival'][station]
                    )
                    
                    # Boolean for whether train stops at this station
                    is_at_station = self.model.NewBoolVar(f'{train_id}_at_{station}')
                    self.model.Add(is_at_station == 1)  # Train always stops at defined stations
                    
                    interval = self.model.NewOptionalIntervalVar(
                        self.variables[train_id]['arrival'][station],
                        size,
                        self.variables[train_id]['departure'][station],
                        is_at_station,
                        f'{train_id}_{station}_platform'
                    )
                    intervals.append(interval)
            
            if len(intervals) > 1:
                self.model.AddNoOverlap(intervals)
    
    def _add_maintenance_constraints(self):
        """Add maintenance window constraints"""
        print("Adding maintenance constraints...")
        
        for maintenance in self.maintenance_windows:
            block_parts = maintenance['block'].split('-')
            if len(block_parts) >= 3:
                station1 = block_parts[0]
                station2 = block_parts[1]
                block_num = block_parts[2]
                segment = f"{station1}-{station2}"
                block_id = f"{segment}_{block_num}"
                
                start_time = maintenance['start_time']
                end_time = maintenance['end_time']
                
                # Find trains that use this block
                for train in self.trains:
                    train_id = train['train']
                    if block_id in self.variables[train_id]['block_entry']:
                        # Train must clear block before maintenance starts or enter after it ends
                        b = self.model.NewBoolVar(f'maintenance_{train_id}_{block_id}')
                        
                        # Either: Train exits block before maintenance starts
                        self.model.Add(
                            self.variables[train_id]['block_exit'][block_id] <= start_time
                        ).OnlyEnforceIf(b)
                        
                        # Or: Train enters block after maintenance ends
                        self.model.Add(
                            self.variables[train_id]['block_entry'][block_id] >= end_time
                        ).OnlyEnforceIf(b.Not())
    
    def _add_operational_constraints(self):
        """Add operational constraints including priority and connections"""
        print("Adding operational constraints...")
        
        # Connection constraints - if trains connect at a station
        # For this example, let's assume express trains connect with local trains at S2
        connection_station = "S2"
        express_trains = [t for t in self.trains if t['type'] == 'express' and connection_station in t['route']]
        local_trains = [t for t in self.trains if t['type'] == 'local' and connection_station in t['route']]
        
        for express in express_trains:
            for local in local_trains:
                if connection_station in express['route'] and connection_station in local['route']:
                    express_idx = express['route'].index(connection_station)
                    local_idx = local['route'].index(connection_station)
                    if abs(express['planned'][express_idx] - local['planned'][local_idx]) < 10:
                        # These trains might connect
                        express_id = express['train']
                        local_id = local['train']
                        
                        # Ensure minimum transfer time
                        b = self.model.NewBoolVar(f'connection_{express_id}_{local_id}')
                        
                        # Express arrives before local departs
                        self.model.Add(
                            self.variables[express_id]['arrival'][connection_station] + self.MIN_TRANSFER_TIME <=
                            self.variables[local_id]['departure'][connection_station]
                        ).OnlyEnforceIf(b)
                        
                        # Or local arrives before express departs
                        self.model.Add(
                            self.variables[local_id]['arrival'][connection_station] + self.MIN_TRANSFER_TIME <=
                            self.variables[express_id]['departure'][connection_station]
                        ).OnlyEnforceIf(b.Not())
    
    def _define_objective(self):
        """Define the objective function to minimize weighted delays"""
        print("Defining objective function...")
        
        total_weighted_delay = []
        
        for train in self.trains:
            train_id = train['train']
            weight = self.PRIORITY_WEIGHTS[train['type']]
            
            for station in train['route']:
                # Penalize lateness more than earliness
                delay_var = self.variables[train_id]['delay'][station]
                
                # Create variables for positive (late) and negative (early) delays
                late_delay = self.model.NewIntVar(0, self.MAX_LATENESS * 2, f'{train_id}_{station}_late')
                early_delay = self.model.NewIntVar(0, self.MAX_EARLINESS, f'{train_id}_{station}_early')
                
                # late_delay = max(0, delay_var)
                self.model.AddMaxEquality(late_delay, [delay_var, 0])
                
                # early_delay = max(0, -delay_var)
                neg_delay = self.model.NewIntVar(-self.MAX_LATENESS * 2, 0, f'{train_id}_{station}_neg')
                self.model.Add(neg_delay == -delay_var)
                self.model.AddMaxEquality(early_delay, [neg_delay, 0])
                
                # Weight: lateness penalty = 2x earliness penalty
                total_weighted_delay.append(weight * (2 * late_delay + early_delay))
        
        # Minimize total weighted delay
        self.model.Minimize(sum(total_weighted_delay))
    
    def solve_model(self):
        """Solve the optimization model"""
        print("\nSolving model...")
        
        self.solver.parameters.max_time_in_seconds = 30.0
        self.solver.parameters.num_search_workers = 4
        
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL:
            print("✓ Optimal solution found!")
        elif status == cp_model.FEASIBLE:
            print("✓ Feasible solution found (may not be optimal)")
        else:
            print("✗ No feasible solution found")
            return False
        
        # Extract solution
        self._extract_solution()
        
        # Print solution details
        self._print_solution()
        
        return True
    
    def _extract_solution(self):
        """Extract solution values from solver"""
        for train in self.trains:
            train_id = train['train']
            self.solution[train_id] = {
                'route': train['route'],
                'type': train['type'],
                'planned': train['planned'],
                'actual_arrival': {},
                'actual_departure': {},
                'delays': {},
                'blocks': {}
            }
            
            for station in train['route']:
                self.solution[train_id]['actual_arrival'][station] = \
                    self.solver.Value(self.variables[train_id]['arrival'][station])
                self.solution[train_id]['actual_departure'][station] = \
                    self.solver.Value(self.variables[train_id]['departure'][station])
                self.solution[train_id]['delays'][station] = \
                    self.solver.Value(self.variables[train_id]['delay'][station])
            
            # Extract block occupancy times
            for block_id in self.variables[train_id]['block_entry']:
                entry_time = self.solver.Value(self.variables[train_id]['block_entry'][block_id])
                exit_time = self.solver.Value(self.variables[train_id]['block_exit'][block_id])
                self.solution[train_id]['blocks'][block_id] = (entry_time, exit_time)
    
    def _print_solution(self):
        """Print the solution details"""
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        
        total_delay = 0
        
        for train_id, sol in self.solution.items():
            print(f"\nTrain {train_id} ({sol['type'].upper()}):")
            print("-" * 40)
            
            for i, station in enumerate(sol['route']):
                planned = sol['planned'][i]
                actual_arr = sol['actual_arrival'][station]
                actual_dep = sol['actual_departure'][station]
                delay = sol['delays'][station]
                
                status = "ON TIME" if abs(delay) <= 1 else f"{'LATE' if delay > 0 else 'EARLY'} by {abs(delay)} min"
                
                print(f"  {station}:")
                print(f"    Planned arrival: {planned:3d} min")
                print(f"    Actual arrival:  {actual_arr:3d} min  [{status}]")
                print(f"    Actual departure:{actual_dep:3d} min")
                
                total_delay += abs(delay) * self.PRIORITY_WEIGHTS[sol['type']]
            
            print(f"\n  Block assignments:")
            for block_id, (entry, exit) in sorted(sol['blocks'].items()):
                print(f"    {block_id}: [{entry:3d} - {exit:3d}] min")
        
        print(f"\n" + "="*80)
        print(f"Total weighted delay: {total_delay}")
        print(f"Objective value: {self.solver.ObjectiveValue()}")
        print("="*80)
    
    def visualize_schedule(self):
        """Create a Gantt chart visualization of the schedule"""
        print("\nGenerating visualization...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Prepare data for plotting
        train_list = list(self.solution.keys())
        y_pos = {train: i for i, train in enumerate(train_list)}
        
        # Color scheme
        type_colors = {
            'express': '#FF6B6B',
            'local': '#4ECDC4',
            'freight': '#95A5A6'
        }
        
        # Plot 1: Station occupancy
        ax1.set_title('Train Schedule - Station Occupancy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('Trains', fontsize=12)
        
        for train_id, sol in self.solution.items():
            y = y_pos[train_id]
            color = type_colors[sol['type']]
            
            for i, station in enumerate(sol['route']):
                # Planned schedule (light/transparent)
                planned_arr = sol['planned'][i]
                planned_dep = planned_arr + self.MIN_DWELL_TIME
                ax1.barh(y, planned_dep - planned_arr, left=planned_arr, 
                        height=0.3, alpha=0.3, color=color, edgecolor='black', linewidth=0.5)
                
                # Actual schedule (solid)
                actual_arr = sol['actual_arrival'][station]
                actual_dep = sol['actual_departure'][station]
                ax1.barh(y, actual_dep - actual_arr, left=actual_arr, 
                        height=0.3, alpha=0.8, color=color, edgecolor='black', linewidth=1)
                
                # Station label
                ax1.text(actual_arr + (actual_dep - actual_arr)/2, y, station, 
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        ax1.set_yticks(range(len(train_list)))
        ax1.set_yticklabels(train_list)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, self.TIME_HORIZON)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=color, alpha=0.3, label=f'{type.capitalize()} (planned)') 
            for type, color in type_colors.items()
        ] + [
            mpatches.Patch(color=color, alpha=0.8, label=f'{type.capitalize()} (actual)') 
            for type, color in type_colors.items()
        ]
        ax1.legend(handles=legend_elements, loc='upper right', ncol=2)
        
        # Plot 2: Block occupancy
        ax2.set_title('Train Schedule - Block Occupancy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_ylabel('Blocks', fontsize=12)
        
        # Collect all blocks
        all_blocks = []
        for segment, blocks in sorted(self.blocks.items()):
            all_blocks.extend(blocks)
        
        block_y_pos = {block: i for i, block in enumerate(all_blocks)}
        
        for train_id, sol in self.solution.items():
            color = type_colors[sol['type']]
            
            for block_id, (entry, exit) in sol['blocks'].items():
                if block_id in block_y_pos:
                    y = block_y_pos[block_id]
                    ax2.barh(y, exit - entry, left=entry, height=0.8, 
                            alpha=0.7, color=color, edgecolor='black', linewidth=1)
                    
                    # Train label on block
                    if exit - entry > 2:  # Only show label if block is wide enough
                        ax2.text(entry + (exit - entry)/2, y, train_id, 
                                ha='center', va='center', fontsize=7, color='white')
        
        # Add maintenance window visualization
        for maintenance in self.maintenance_windows:
            block_parts = maintenance['block'].split('-')
            if len(block_parts) >= 3:
                station1 = block_parts[0]
                station2 = block_parts[1]
                block_num = block_parts[2]
                segment = f"{station1}-{station2}"
                block_id = f"{segment}_{block_num}"
                
                if block_id in block_y_pos:
                    y = block_y_pos[block_id]
                    start_time = maintenance['start_time']
                    end_time = maintenance['end_time']
                    ax2.barh(y, end_time - start_time, left=start_time, height=0.8, 
                            alpha=0.5, color='red', edgecolor='red', linewidth=2, hatch='///')
                    ax2.text(start_time + (end_time - start_time)/2, y, 'MAINT', 
                            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        ax2.set_yticks(range(len(all_blocks)))
        ax2.set_yticklabels([b.split('_')[-1] if len(b.split('_')[-1]) <= 3 else b for b in all_blocks], 
                            fontsize=8)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, self.TIME_HORIZON)
        
        # Add segment labels
        y_offset = -1
        for segment in sorted(self.blocks.keys()):
            blocks = self.blocks[segment]
            if blocks:
                first_block_y = block_y_pos[blocks[0]]
                last_block_y = block_y_pos[blocks[-1]]
                ax2.text(-5, (first_block_y + last_block_y) / 2, segment, 
                        ha='right', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('schedule.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved as 'schedule.png'")
        plt.show()


def main():
    """Main function to run the train dispatch optimization"""
    print("=" * 80)
    print("TRAIN DISPATCH OPTIMIZATION SYSTEM")
    print("Using Google OR-Tools CP-SAT Solver")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = TrainDispatchOptimizer()
    
    # Load data
    optimizer.load_data('trains.json')
    optimizer.load_maintenance_data('maintenance.json')
    
    # Build model
    optimizer.build_model()
    
    # Solve model
    if optimizer.solve_model():
        # Visualize results
        optimizer.visualize_schedule()
    else:
        print("Failed to find a solution. Consider relaxing constraints.")


if __name__ == "__main__":
    main()