ALTER TABLE devices
    ADD COLUMN monitoring_state TEXT NOT NULL DEFAULT 'unknown'
        CHECK (monitoring_state IN ('active', 'muted', 'unknown')),
    ADD COLUMN monitoring_updated_at TIMESTAMPTZ;

-- Existing messages also establish when this device last communicated with us.
UPDATE devices d SET last_seen_at = latest.received_at
FROM (SELECT device_id, max(received_at) AS received_at FROM events GROUP BY device_id) latest
WHERE d.device_id = latest.device_id
    AND (d.last_seen_at IS NULL OR d.last_seen_at < latest.received_at);

-- Recover the last explicit monitoring report. Queue delays cannot make an old
-- mute/unmute report newer than a current one. Exclude extreme ages that cannot
-- describe a current monitoring state or safely fit into timestamp arithmetic.
WITH reports AS (
    SELECT device_id,
        CASE event_type WHEN 'device_muted' THEN 'muted'
                        WHEN 'device_unmuted' THEN 'active'
                        ELSE details->>'monitoring_state' END AS state,
        LEAST(detected_at, received_at - elapsed_seconds * interval '1 second') AS reported_at
    FROM events
    WHERE elapsed_seconds BETWEEN 0 AND 3155760000
      AND (event_type IN ('device_muted', 'device_unmuted')
           OR (event_type = 'heartbeat' AND details->>'monitoring_state' IN ('active', 'muted', 'unknown')))
), latest AS (
    SELECT DISTINCT ON (device_id) device_id, state, reported_at FROM reports
    -- Conflicting reports with an identical time resolve conservatively to muted.
    ORDER BY device_id, reported_at DESC,
        CASE state WHEN 'muted' THEN 2 WHEN 'unknown' THEN 1 ELSE 0 END DESC
)
UPDATE devices d SET monitoring_state = latest.state, monitoring_updated_at = latest.reported_at
FROM latest WHERE d.device_id = latest.device_id;
