
"""Pure-Python SheetSage2 rhythm-event analysis.

This helper analyses the event list returned by SheetSage2.transcribe().
It does not load models, process audio, change ABC, or run transcription.
"""


from collections import Counter
from statistics import median
from .rhythm_report_helper import analyze_events


def _meter(value):
    """Convert (4, 4), [4, 4], or '4/4' into a validated tuple."""
    try:
        if isinstance(value, str):
            a, b = value.strip().split("/", 1)
            value = (int(a), int(b))

        if not isinstance(value, (tuple, list)) or len(value) < 2:
            return None

        a, b = int(value[0]), int(value[1])
        return (a, b) if a > 0 and b > 0 else None

    except (TypeError, ValueError, OverflowError):
        return None


def analyze_events(events):
    """Return rhythm metrics and a readable report from SheetSage2 events."""

    events = list(events) if not isinstance(events, (list, tuple)) else events

    active_meter = None
    explicit_meter_events = 0
    inherited_meter_events = 0
    invalid_meter_values = 0
    rhythm_events = []

    # SheetSage2 may provide the meter only at the beginning.
    # Carry it forward for subsequent events that omit it.
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue

        values = event.get("values")
        if not isinstance(values, dict):
            values = {}

        rhythm = values.get("rhythm", event.get("rhythm", {}))
        if not isinstance(rhythm, dict):
            continue

        if "meter" in rhythm:
            parsed_meter = _meter(rhythm["meter"])
            if parsed_meter is not None:
                active_meter = parsed_meter
            else:
                invalid_meter_values += 1

        if active_meter is None or "eighth_position" not in rhythm:
            continue

        try:
            position = int(rhythm["eighth_position"])
        except (TypeError, ValueError, OverflowError):
            continue

        if "meter" in rhythm and _meter(rhythm["meter"]) is not None:
            explicit_meter_events += 1
        else:
            inherited_meter_events += 1

        timestamp = event.get("time", values.get("timestamp"))

        try:
            timestamp = float(timestamp)
            if timestamp < 0:
                continue
        except (TypeError, ValueError, OverflowError):
            continue

        rhythm_events.append({
            "index": index,
            "time": timestamp,
            "meter": active_meter,
            "position": position,
        })

    rhythm_events.sort(key=lambda item: (item["time"], item["index"]))

    meter_counts = Counter(item["meter"] for item in rhythm_events)
    position_counts = Counter(
        item["position"] for item in rhythm_events
    )

    # Unwrap eighth_position at bar boundaries, then convert positions
    # into denominator-note beats. In 4/4 these are quarter-note beats.
    points = []
    bar_offset = 0.0
    previous_position = None
    previous_meter = None

    for item in rhythm_events:
        numerator, denominator = item["meter"]
        position = item["position"]

        if (
            previous_meter is not None
            and item["meter"] != previous_meter
        ):
            bar_offset = 0.0

        elif (
            previous_position is not None
            and position < previous_position
        ):
            bar_offset += numerator * 8.0 / denominator

        beat_position = (
            bar_offset + position
        ) * denominator / 8.0

        points.append({
            **item,
            "beat": beat_position,
        })

        previous_position = position
        previous_meter = item["meter"]

    # Estimate tempo using both elapsed time and the change in beat position.
    # This avoids assuming that every pair of records is exactly one beat apart.
    seconds_per_beat = []
    intervals_skipped = 0

    for left, right in zip(points, points[1:]):
        elapsed = right["time"] - left["time"]
        beat_delta = right["beat"] - left["beat"]

        if (
            elapsed <= 0
            or beat_delta <= 0
            or left["meter"] != right["meter"]
        ):
            intervals_skipped += 1
            continue

        interval = elapsed / beat_delta
        bpm = 60.0 / interval

        # Exclude implausible intervals rather than letting them distort
        # the summary. This is a diagnostic estimate, not a tempo tracker.
        if 30.0 <= bpm <= 300.0:
            seconds_per_beat.append(interval)
        else:
            intervals_skipped += 1

    median_interval = (
        median(seconds_per_beat) if seconds_per_beat else None
    )

    mean_interval = (
        sum(seconds_per_beat) / len(seconds_per_beat)
        if seconds_per_beat else None
    )

    median_bpm = (
        60.0 / median_interval if median_interval else None
    )

    mean_bpm = (
        60.0 / mean_interval if mean_interval else None
    )

    dominant_meter = (
        meter_counts.most_common(1)[0][0]
        if meter_counts else None
    )

    meter_summary = ", ".join(
        f"{n}/{d}: {count} rhythm events"
        for (n, d), count in sorted(
            meter_counts.items(),
            key=lambda pair: (-pair[1], pair[0])
        )
    ) or "unavailable"

    lines = [
        "SHEETSAGE2 RHYTHM REPORT",
        f"Input event records: {len(events)}",
        f"Timed rhythm events: {len(rhythm_events)}",
        f"Explicit meter observations: {explicit_meter_events}",
        f"Events inheriting meter: {inherited_meter_events}",
        f"Invalid meter values: {invalid_meter_values}",
        f"Meter coverage: {meter_summary}",
        (
            f"Dominant meter: {dominant_meter[0]}/{dominant_meter[1]}"
            if dominant_meter else "Dominant meter: unavailable"
        ),
        "eighth_position counts: " + (
            ", ".join(
                f"{p}={c}" for p, c in sorted(position_counts.items())
            ) or "unavailable"
        ),
    ]

    if median_bpm is not None:
        lines.extend([
            f"Median interval-based tempo: {median_bpm:.2f} BPM",
            f"Mean interval-based tempo: {mean_bpm:.2f} BPM",
            f"Intervals used: {len(seconds_per_beat)}",
            f"Intervals skipped: {intervals_skipped}",
            f"Median seconds per beat: {median_interval:.4f}",
        ])
    else:
        lines.extend([
            "Tempo: unavailable (insufficient valid intervals)",
            f"Intervals skipped: {intervals_skipped}",
        ])

    lines.extend([
        "",
        "INTERPRETATION NOTES",
        "- Meter is carried forward when later events omit it.",
        "- Tempo accounts for changes in symbolic beat position.",
        "- This measures the symbolic beat grid, not kick/snare accents.",
        "- Half-time/double-time ambiguity may remain.",
    ])

    return {
        "event_count": len(events),
        "timed_rhythm_event_count": len(rhythm_events),
        "explicit_meter_observations": explicit_meter_events,
        "inherited_meter_events": inherited_meter_events,
        "invalid_meter_values": invalid_meter_values,
        "meter_counts": {
            f"{n}/{d}": count
            for (n, d), count in meter_counts.items()
        },
        "dominant_meter": (
            f"{dominant_meter[0]}/{dominant_meter[1]}"
            if dominant_meter else None
        ),
        "position_counts": dict(sorted(position_counts.items())),
        "median_bpm": median_bpm,
        "mean_bpm": mean_bpm,
        "median_seconds_per_beat": median_interval,
        "intervals_used": len(seconds_per_beat),
        "intervals_skipped": intervals_skipped,
        "text": "\n".join(lines),
    }


if __name__ == "__main__":
    # Smoke test: 4/4 at 123 BPM, with meter supplied only once.
    interval = 60.0 / 123.0
    test_events = []

    for i, position in enumerate([0, 2, 4, 6, 0, 2, 4, 6, 0]):
        rhythm = {"eighth_position": position}

        if i == 0:
            rhythm["meter"] = (4, 4)

        test_events.append({
            "time": i * interval,
            "values": {"rhythm": rhythm},
        })

    result = analyze_events(test_events)
    print(result["text"])

    assert result["dominant_meter"] == "4/4"
    assert abs(result["median_bpm"] - 123.0) < 0.1