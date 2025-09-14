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

// ------------------------------
// Insert trains
// ------------------------------
db.trains.insertMany([
  { train_id: "T001", type: "passenger", priority: 1, avg_speed_kmh: 100, length_m: 200 },
  { train_id: "T002", type: "freight",   priority: 3, avg_speed_kmh: 60,  length_m: 500 },
  { train_id: "T003", type: "express",   priority: 0, avg_speed_kmh: 120, length_m: 180 }
])

// ------------------------------
// Insert stations
// ------------------------------
db.stations.insertMany([
  { _id: "S1", name: "Central",  platforms: 4 },
  { _id: "S2", name: "WestSide", platforms: 2 },
  { _id: "S3", name: "EastEnd",  platforms: 3 }
])

// ------------------------------
// Insert segments
// ------------------------------
db.segments.insertMany([
  { _id: "seg_S1_S2", from: "S1", to: "S2", capacity: 1, travel_time_min: 20 },
  { _id: "seg_S2_S3", from: "S2", to: "S3", capacity: 1, travel_time_min: 25 }
])

// ------------------------------
// Insert timetable with events
// ------------------------------
db.timetable.insertMany([
  {
    train_id: "T001",
    events: [
      {
        event_id: "E1",
        type: "departure",
        station_id: "S1",
        scheduled_time: ISODate("2025-09-20T08:00:00Z"),
        earliness_sec: 60,
        lateness_sec: 300,
        min_dwell_sec: 120
      },
      {
        event_id: "E2",
        type: "arrival",
        station_id: "S2",
        scheduled_time: ISODate("2025-09-20T08:20:00Z"),
        earliness_sec: 60,
        lateness_sec: 300
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
        scheduled_time: ISODate("2025-09-20T08:10:00Z"),
        earliness_sec: 120,
        lateness_sec: 600
      },
      {
        event_id: "E2",
        type: "arrival",
        station_id: "S3",
        scheduled_time: ISODate("2025-09-20T08:35:00Z"),
        earliness_sec: 120,
        lateness_sec: 600
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
  }
])

// ------------------------------
// Insert scenario manifest
// ------------------------------
db.scenarios.insertOne({
  _id: "scenario_01",
  description: "Morning traffic with mixed freight/passenger",
  trains: ["T001", "T002"],
  segments: ["seg_S1_S2", "seg_S2_S3"],
  constraints: ["maintenance", "headway"]
})

// ------------------------------
// Ensure unique index for merge
// ------------------------------
db.train_events.createIndex({ train_id: 1, event_id: 1 }, { unique: true })

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
      segment_id: "$events.segment_id",
      scheduled_time: "$events.scheduled_time",
      earliness_sec: "$events.earliness_sec",
      lateness_sec: "$events.lateness_sec",
      min_dwell_sec: "$events.min_dwell_sec"
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
// Create indexes for performance
// ------------------------------
db.train_events.createIndex({ scheduled_time: 1 })
db.segments.createIndex({ from: 1, to: 1 })
db.constraints.createIndex({ segment_id: 1, start: 1, end: 1 })

print("railwayDB setup complete")






//TO RUN---->
//connect
//use railwayDB
//cls
