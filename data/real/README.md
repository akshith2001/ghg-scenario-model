# Building-data extract

The CSV contains real BDG2 building electricity data. Its buildings are at different
sites and are mapped onto a simulated feeder for these experiments. Daily mean power
cannot capture within-day voltage peaks or hourly losses; the daily feeder calculation
is an approximation, not measured grid consumption.

The processed CSV hash in the provenance JSON is for LF-normalised bytes. On Windows,
CRLF line endings produce a different byte hash without changing CSV values. Corrected
experiment manifests record the exact byte hash used for each run.

The data distribution's licence and attribution are documented in the provenance JSON;
the repository's Apache licence does not replace third-party data terms.
