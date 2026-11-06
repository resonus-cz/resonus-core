---
format: splmeter-session
version: "1.2"
name: "Office: \"north\" #2"
start_utc: "2026-11-05T20:40:42.515Z"
end_utc: "2026-11-05T20:40:51.515Z"
duration_s: 9.000
software: "SPL Meter 0.1.0"
log_interval_s: 1.000
reference_pressure_uPa: 20
device:
  id: "demo.sine-1khz"
  name: "Demo sine source (1 kHz)"
  backend: "demo"
  channel: 1
  sample_rate_hz: 48000
calibration:
  offset_db: 0.00
  type: none
tags:
  - "demo"
rta:
  quantity: LZeq
  bands_per_octave: 3
  bands_hz: [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
files:
  log: log.csv
  markers: markers.csv
  segments: segments.csv
  rta_log: rta_log.csv
  rta_segments: rta_segments.csv
---

# Office: "north" #2

## Description

Line one.

- item

## Summary

```yaml {#summary}
range: { start_s: 0.000, end_s: 9.000 }
A: { Leq_db: -23.01, peak_db: -19.83, Fmax_db: -23.01 }
C: { Leq_db: -23.01, peak_db: -19.69, Fmax_db: -23.01 }
Z: { Leq_db: -23.01, peak_db: -19.96, Fmax_db: -23.01 }
rta_LZeq_db: [-93.58, -92.58, -91.57, -90.56, -89.55, -88.53, -87.50, -86.44, -85.36, -84.22, -82.99, -81.61, -78.20, -72.21, -61.25, -41.47, -23.01, -41.04, -59.72, -71.01, -79.28, -86.45, -91.81, -96.18, -100.03, -103.19, -105.97]
clip: false
```
