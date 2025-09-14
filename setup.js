// mongo setup script
// setup.js
// Run with: mongosh < setup.js

// ------------------------------
// Select database
// ------------------------------
db = db.getSiblingDB("railwayDB");

// ------------------------------
// Drop existing collections (for clean re-run)
// ------------------------------
db.trains.drop()
db.stations.drop()
db.segments.drop()
db.timetable.drop()
db.constraints.drop()
db.scenarios.drop()
db.train_events.drop()
db.platform_occupancy.drop() // New collection for platform tracking

// ------------------------------
// Create collections
// ------------------------------
db.createCollection("trains")
db.createCollection("stations")
db.createCollection("segments")
db.createCollection("timetable")
db.createCollection("constraints")
db.createCollection("scenarios")
db.createCollection("train_events")
db.createCollection("platform_occupancy") // New collection

// ------------------------------
// Insert trains
// ------------------------------
db.trains.insertMany([
  { train_id: "T001", type: "passenger", priority: 1, avg_speed_kmh: 100, length_m: 200 },
  { train_id: "T002", type: "freight",   priority: 3, avg_speed_kmh: 60,  length_m: 500 },
  { train_id: "T003", type: "express",   priority: 0, avg_speed_kmh: 120, length_m: 180 }
])

// ------------------------------
// Insert stations with detailed platform information
// ------------------------------
db.stations.insertMany([
  { 
    _id: "S1", 
    name: "Central", 
    total_platforms: 4,
    platforms: [
      { platform_id: "S1P1", length_m: 250, electrified: true },
      { platform_id: "S1P2", length_m: 250, electrified: true },
      { platform_id: "S1P3", length_m: 300, electrified: true },
      { platform_id: "S1P4", length_m: 350, electrified: false }
    ]
  },
  { 
    _id: "S2", 
    name: "WestSide", 
    total_platforms: 2,
    platforms: [
      { platform_id: "S2P1", length_m: 200, electrified: true },
      { platform_id: "S2P2", length_m: 250, electrified: false }
    ]
  },
  { 
    _id: "S3", 
    name: "EastEnd", 
    total_platforms: 3,
    platforms: [
      { platform_id: "S3P1", length_m: 220, electrified: true },
      { platform_id: "S3P2", length_m: 220, electrified: true },
      { platform_id: "S3P3", length_m: 300, electrified: false }
    ]
  }
])

// ------------------------------
// Insert segments
// ------------------------------
db.segments.insertMany([
  { _id: "seg_S1_S2", from: "S1", to: "S2", capacity: 1, travel_time_min: 20 },
  { _id: "seg_S2_S3", from: "S2", to: "S3", capacity: 1, travel_time_min: 25 }
])

// ------------------------------
// Insert timetable with events and platform assignments
// ------------------------------
db.timetable.insertMany([
  {
    train_id: "T001",
    events: [
      {
        event_id: "E1",
        type: "departure",
        station_id: "S1",
        platform_id: "S1P2",  // Assigned platform
        scheduled_time: ISODate("2025-09-20T08:00:00Z"),
        earliness_sec: 60,
        lateness_sec: 300,
        min_dwell_sec: 120
      },
      {
        event_id: "E2",
        type: "arrival",
        station_id: "S2",
        platform_id: "S2P1",  // Assigned platform
        scheduled_time: ISODate("2025-09-20T08:20:00Z"),
        earliness_sec: 60,
        lateness_sec: 300,
        dwell_time_sec: 300  // How long the train stays at platform after arrival
      }
    ]
  },
  {
    train_id: "T002",
    events: [
      {
        event_id: "E1",
        type: "departure",
        station_id: "S2",
        platform_id: "S2P2",  // Assigned platform
        scheduled_time: ISODate("2025-09-20T08:10:00Z"),
        earliness_sec: 120,
        lateness_sec: 600,
        min_dwell_sec: 180
      },
      {
        event_id: "E2",
        type: "arrival",
        station_id: "S3",
        platform_id: "S3P3",  // Assigned platform
        scheduled_time: ISODate("2025-09-20T08:35:00Z"),
        earliness_sec: 120,
        lateness_sec: 600,
        dwell_time_sec: 600  // How long the train stays at platform after arrival
      }
    ]
  },
  {
    train_id: "T003",
    events: [
      {
        event_id: "E1",
        type: "departure",
        station_id: "S1",
        platform_id: "S1P1",  // Assigned platform
        scheduled_time: ISODate("2025-09-20T08:15:00Z"),
        earliness_sec: 60,
        lateness_sec: 180,
        min_dwell_sec: 90
      },
      {
        event_id: "E2",
        type: "arrival",
        station_id: "S2",
        platform_id: "S2P1",  // Assigned platform
        scheduled_time: ISODate("2025-09-20T08:35:00Z"),
        earliness_sec: 60,
        lateness_sec: 180,
        dwell_time_sec: 240  // How long the train stays at platform after arrival
      }
    ]
  }
])

// ------------------------------
// Insert constraints
// ------------------------------
db.constraints.insertMany([
  {
    type: "maintenance",
    segment_id: "seg_S1_S2",
    start: ISODate("2025-09-20T07:30:00Z"),
    end:   ISODate("2025-09-20T07:50:00Z"),
    description: "Track closed for inspection"
  },
  {
    type: "headway",
    segment_id: "seg_S2_S3",
    min_gap_sec: 300
  },
  {
    type: "platform_maintenance",
    station_id: "S1",
    platform_id: "S1P3",
    start: ISODate("2025-09-20T08:00:00Z"),
    end: ISODate("2025-09-20T10:00:00Z"),
    description: "Platform renovation"
  }
])

// ------------------------------
// Insert scenario manifest
// ------------------------------
db.scenarios.insertOne({
  _id: "scenario_01",
  description: "Morning traffic with mixed freight/passenger",
  trains: ["T001", "T002", "T003"],
  segments: ["seg_S1_S2", "seg_S2_S3"],
  constraints: ["maintenance", "headway", "platform_maintenance"]
})

// ------------------------------
// Ensure unique indices
// ------------------------------
db.train_events.createIndex({ train_id: 1, event_id: 1 }, { unique: true })
db.platform_occupancy.createIndex({ platform_id: 1, start_time: 1, end_time: 1 }, { unique: false })

// ------------------------------
// Populate train_events by flattening timetable
// ------------------------------
db.timetable.aggregate([
  { $unwind: "$events" },
  {
    $project: {
      _id: 0,  // don't reuse timetable _id
      train_id: 1,
      event_id: "$events.event_id",
      type: "$events.type",
      station_id: "$events.station_id",
      platform_id: "$events.platform_id",  // Include platform ID
      segment_id: "$events.segment_id",
      scheduled_time: "$events.scheduled_time",
      earliness_sec: "$events.earliness_sec",
      lateness_sec: "$events.lateness_sec",
      min_dwell_sec: "$events.min_dwell_sec",
      dwell_time_sec: "$events.dwell_time_sec"
    }
  },
  {
    $merge: {
      into: "train_events",
      on: ["train_id", "event_id"],
      whenMatched: "replace",
      whenNotMatched: "insert"
    }
  }
])

// ------------------------------
// Create platform occupancy records based on train events
// ------------------------------

// First, let's get all arrival events
let arrivalEvents = db.train_events.find({type: "arrival"}).toArray();

// For each arrival, create a platform occupancy record
arrivalEvents.forEach(arrival => {
  // Calculate departure time based on dwell time
  let departureTime = new Date(arrival.scheduled_time);
  departureTime.setSeconds(departureTime.getSeconds() + (arrival.dwell_time_sec || 300)); // Default 5 min if not specified
  
  // Get train details
  let train = db.trains.findOne({train_id: arrival.train_id});
  
  db.platform_occupancy.insertOne({
    train_id: arrival.train_id,
    train_type: train.type,
    train_length_m: train.length_m,
    station_id: arrival.station_id,
    platform_id: arrival.platform_id,
    start_time: arrival.scheduled_time,
    end_time: departureTime,
    duration_sec: arrival.dwell_time_sec || 300
  });
});

// Also handle departure events (trains are at the platform before departure)
let departureEvents = db.train_events.find({type: "departure"}).toArray();

departureEvents.forEach(departure => {
  // Calculate arrival time based on min_dwell_sec
  let arrivalTime = new Date(departure.scheduled_time);
  arrivalTime.setSeconds(arrivalTime.getSeconds() - (departure.min_dwell_sec || 120)); // Default 2 min if not specified
  
  // Get train details
  let train = db.trains.findOne({train_id: departure.train_id});
  
  db.platform_occupancy.insertOne({
    train_id: departure.train_id,
    train_type: train.type,
    train_length_m: train.length_m,
    station_id: departure.station_id,
    platform_id: departure.platform_id,
    start_time: arrivalTime,
    end_time: departure.scheduled_time,
    duration_sec: departure.min_dwell_sec || 120
  });
});

// ------------------------------
// Create indices for performance
// ------------------------------
db.train_events.createIndex({ scheduled_time: 1 })
db.train_events.createIndex({ station_id: 1, platform_id: 1 })
db.segments.createIndex({ from: 1, to: 1 })
db.constraints.createIndex({ segment_id: 1, start: 1, end: 1 })
db.platform_occupancy.createIndex({ station_id: 1 })
db.platform_occupancy.createIndex({ platform_id: 1 })
db.platform_occupancy.createIndex({ start_time: 1, end_time: 1 })

print("railwayDB setup complete with platform tracking")
