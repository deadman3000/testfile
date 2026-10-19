
"""Pure-Python SheetSage2 rhythm-event analysis.

This helper analyses the event list returned by SheetSage2.transcribe().
It does not load models, process audio, change ABC, or run transcription.
"""


from collections import Counter
from statistics import median


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

def analyze_audio_onsets(waveform, sample_rate, bpm, beat_offset=0.0):
    """
    Analyse audio spectral flux, estimate beat phase, compare quarter-note
    and eighth-note positions, and measure bar-to-bar profile variation.

    CPU-only analysis. Does not modify the waveform or require new packages.
    """
    import math
    import torch
    import torch.nn.functional as F

    lines = [
        "AUDIO ONSET / ACCENT ANALYSIS",
        "Method: CPU spectral flux; this is not instrument separation.",
    ]

    try:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        audio = torch.as_tensor(waveform).detach().to(
            device="cpu", dtype=torch.float32
        )

        if audio.ndim == 3:
            audio = audio[0].mean(dim=0)
        elif audio.ndim == 2:
            audio = audio.mean(dim=0)
        elif audio.ndim != 1:
            raise ValueError(
                f"Unsupported waveform shape: {list(audio.shape)}"
            )

        if audio.numel() < 2048:
            raise ValueError("Audio is too short for onset analysis.")

        audio = torch.nan_to_num(audio)
        duration = audio.numel() / sample_rate

        # Generate spectral flux at approximately 100 frames per second.
        n_fft = 2048
        hop_length = max(1, round(sample_rate * 0.010))
        hop_seconds = hop_length / sample_rate
        window = torch.hann_window(n_fft)

        with torch.inference_mode():
            spectrum = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )

            magnitude = spectrum.abs()

            # Ignore DC; measure only positive changes in spectral energy.
            difference = magnitude[1:, 1:] - magnitude[1:, :-1]
            flux = difference.clamp_min(0).mean(dim=0)

            if flux.numel() >= 3:
                flux = F.avg_pool1d(
                    flux.reshape(1, 1, -1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).reshape(-1)

        del spectrum, magnitude, difference

        if flux.numel() == 0 or not bool(torch.isfinite(flux).all()):
            raise ValueError("Spectral-flux calculation was invalid.")

        flux = torch.nan_to_num(flux).clamp_min(0)
        frame_count = flux.numel()
        peak_flux = float(flux.max())
        mean_flux = float(flux.mean())
        median_flux = float(torch.median(flux))

        lines.extend([
            f"Audio duration: {duration:.2f} seconds",
            f"Analysis hop: {hop_seconds * 1000:.2f} ms",
            f"Mean spectral flux: {mean_flux:.6g}",
            f"Median spectral flux: {median_flux:.6g}",
            f"Maximum spectral flux: {peak_flux:.6g}",
        ])

        # Rough onset count; this does not represent note or drum-hit count.
        onset_count = 0

        if frame_count >= 3 and peak_flux > 0:
            threshold = float(torch.quantile(flux, 0.75))
            peaks = (
                (flux[1:-1] >= flux[:-2])
                & (flux[1:-1] > flux[2:])
                & (flux[1:-1] >= threshold)
            )
            peak_indices = (torch.where(peaks)[0] + 1).tolist()

            min_separation = max(1, round(0.080 / hop_seconds))
            last_index = -min_separation

            for index in peak_indices:
                if index - last_index >= min_separation:
                    onset_count += 1
                    last_index = index

        onset_rate = onset_count / duration

        lines.extend([
            f"Prominent onset candidates: {onset_count}",
            f"Onset candidate rate: {onset_rate:.2f} per second",
            (
                "Onset candidates can include vocals, melody, cymbals, "
                "chord changes, percussion, and other transients."
            ),
        ])

        try:
            bpm = float(bpm)
        except (TypeError, ValueError, OverflowError):
            bpm = None

        if bpm is None or not math.isfinite(bpm) or not 30 <= bpm <= 300:
            lines.extend([
                "",
                "BEAT-GRID ANALYSIS",
                "Unavailable: no valid symbolic BPM estimate.",
            ])
            return {
                "text": "\n".join(lines),
                "onset_count": onset_count,
                "onset_rate": onset_rate,
                "mean_spectral_flux": mean_flux,
                "median_spectral_flux": median_flux,
            }

        beat_period = 60.0 / bpm
        eighth_period = beat_period / 2.0

        # Use a narrow window during phase estimation so that the search
        # does not simply benefit from the previous wide +/-50 ms window.
        phase_tolerance = min(0.020, hop_seconds * 2.0)
        measurement_tolerance = min(0.040, beat_period * 0.08)

        try:
            beat_offset = float(beat_offset)
        except (TypeError, ValueError, OverflowError):
            beat_offset = 0.0

        if not math.isfinite(beat_offset):
            beat_offset = 0.0

        beat_offset %= beat_period

        def local_strength(time_seconds, tolerance):
            """Maximum flux near a requested time, or None out of range."""
            if time_seconds < 0 or time_seconds >= duration:
                return None

            start = max(0.0, time_seconds - tolerance)
            end = min(duration, time_seconds + tolerance)

            first = max(0, int(math.floor(start / hop_seconds)))
            last = min(
                frame_count,
                int(math.ceil(end / hop_seconds)) + 1,
            )

            if first >= last:
                return None

            return float(flux[first:last].max())

        # Search one complete quarter-note period for the phase that gives
        # the strongest average response at quarter-note beat positions.
        phase_step = hop_seconds
        phase_count = max(
            1, int(math.ceil(beat_period / phase_step))
        )
        phase_results = []

        for phase_index in range(phase_count):
            phase = phase_index * beat_period / phase_count
            scores = []
            beat_number = 0

            while True:
                time_seconds = phase + beat_number * beat_period
                if time_seconds >= duration:
                    break

                score = local_strength(time_seconds, phase_tolerance)
                if score is not None:
                    scores.append(score)

                beat_number += 1

            if scores:
                phase_results.append({
                    "phase": phase,
                    "score": sum(scores) / len(scores),
                    "count": len(scores),
                })

        if not phase_results:
            raise ValueError("No valid beat phases could be evaluated.")

        phase_results.sort(key=lambda item: item["score"])
        best_phase_result = phase_results[-1]
        best_phase = best_phase_result["phase"]
        best_phase_score = best_phase_result["score"]
        median_phase_score = phase_results[len(phase_results) // 2]["score"]

        phase_improvement = (
            (best_phase_score / median_phase_score - 1.0) * 100.0
            if median_phase_score > 0 else 0.0
        )

        # Measure the selected grid at a slightly wider tolerance.
        # Group quarter-note beats into positions 1-4 and eighth notes
        # into positions 1-8 within each four-quarter-note bar.
        beat_scores = [[] for _ in range(4)]
        eighth_scores = [[] for _ in range(8)]
        bars = {}
        beat_number = 0

        while True:
            beat_time = best_phase + beat_number * beat_period
            if beat_time >= duration:
                break

            beat_class = beat_number % 4
            bar_number = beat_number // 4

            score = local_strength(beat_time, measurement_tolerance)
            if score is not None:
                beat_scores[beat_class].append(score)
                eighth_scores[beat_class * 2].append(score)

            bar_profile = bars.setdefault(bar_number, [None] * 8)
            bar_profile[beat_class * 2] = score

            eighth_time = beat_time + eighth_period
            if eighth_time < duration:
                eighth_score = local_strength(
                    eighth_time, measurement_tolerance
                )
                eighth_scores[beat_class * 2 + 1].append(
                    eighth_score
                ) if eighth_score is not None else None
                bar_profile[beat_class * 2 + 1] = eighth_score

            beat_number += 1

        beat_means = [
            sum(values) / len(values) if values else None
            for values in beat_scores
        ]

        eighth_means = [
            sum(values) / len(values) if values else None
            for values in eighth_scores
        ]

        # Compare the selected quarter-note phase against the halfway
        # positions between beats.
        halfbeat_mean = (
            sum(value for values in eighth_scores[1::2] for value in values)
            / sum(len(values) for values in eighth_scores[1::2])
            if any(eighth_scores[1::2]) else None
        )

        available_beat_means = [
            value for value in beat_means if value is not None
        ]
        beat_reference = (
            sum(available_beat_means) / len(available_beat_means)
            if available_beat_means else 0.0
        )

        available_eighth_means = [
            value for value in eighth_means if value is not None
        ]
        eighth_reference = (
            sum(available_eighth_means) / len(available_eighth_means)
            if available_eighth_means else 0.0
        )

        lines.extend([
            "",
            "BEAT-GRID PHASE SEARCH",
            f"Reference tempo: {bpm:.2f} BPM",
            f"Quarter-note period: {beat_period:.4f} seconds",
            f"Phase candidates tested: {len(phase_results)}",
            f"Best estimated phase: {best_phase:.3f} seconds",
            f"Phase-search score: {best_phase_score:.6g}",
            f"Median phase score: {median_phase_score:.6g}",
            f"Best-vs-median score improvement: {phase_improvement:.1f}%",
            f"Phase-search tolerance: +/-{phase_tolerance * 1000:.0f} ms",
            f"Measurement tolerance: +/-{measurement_tolerance * 1000:.0f} ms",
        ])

        if phase_improvement < 3.0:
            lines.append(
                "Phase confidence note: the best phase barely exceeds "
                "the median tested phase; alignment may be ambiguous."
            )
        elif phase_improvement < 10.0:
            lines.append(
                "Phase confidence note: some phase preference was found, "
                "but the result is not decisive."
            )
        else:
            lines.append(
                "Phase confidence note: the best phase is measurably "
                "stronger than the median tested phase; this is not proof "
                "that it matches the musical downbeat."
            )

        lines.extend([
            "",
            "QUARTER-NOTE BEAT POSITIONS",
            "Mean spectral flux near each beat class:",
        ])

        for index, score in enumerate(beat_means, start=1):
            if score is None:
                lines.append(f"  Beat {index}: unavailable")
            else:
                relative = (
                    score / beat_reference if beat_reference > 0 else 0.0
                )
                lines.append(
                    f"  Beat {index}: {score:.6g} "
                    f"(relative {relative:.2f}x)"
                )

        lines.extend([
            "",
            "EIGHTH-NOTE SUBDIVISIONS",
            "Position 1 is the beat; position 2 is the following half-beat.",
        ])

        for index, score in enumerate(eighth_means, start=1):
            if score is None:
                lines.append(f"  Position {index}: unavailable")
            else:
                relative = (
                    score / eighth_reference
                    if eighth_reference > 0 else 0.0
                )
                lines.append(
                    f"  Position {index}: {score:.6g} "
                    f"(relative {relative:.2f}x)"
                )

        if halfbeat_mean is not None:
            lines.append(
                f"Mean half-beat activity: {halfbeat_mean:.6g}"
            )

        # Assess how much the normalised eight-position profile varies
        # across complete bars. Lower deviation means greater similarity.
        complete_profiles = []

        for profile in bars.values():
            if any(value is None for value in profile):
                continue

            profile_mean = sum(profile) / len(profile)
            if profile_mean > 0:
                complete_profiles.append([
                    value / profile_mean for value in profile
                ])

        lines.extend([
            "",
            "BAR-TO-BAR CONSISTENCY",
            f"Bars with all eight positions measured: {len(complete_profiles)}",
        ])

        profile_deviation = None

        if len(complete_profiles) >= 2:
            average_profile = [
                sum(profile[position] for profile in complete_profiles)
                / len(complete_profiles)
                for position in range(8)
            ]

            profile_deviation = sum(
                abs(profile[position] - average_profile[position])
                for profile in complete_profiles
                for position in range(8)
            ) / (len(complete_profiles) * 8)

            lines.append(
                "Mean absolute deviation of normalised bar profiles: "
                f"{profile_deviation:.4f}"
            )
            lines.append(
                "Lower deviation means the measured eight-position "
                "activity pattern is more similar from bar to bar."
            )
        else:
            lines.append(
                "Insufficient complete bars to estimate bar-to-bar variation."
            )

        lines.extend([
            "",
            "INTERPRETATION",
            "- Phase is estimated from spectral activity, not a known downbeat.",
            "- A phase search can align with repeated melody or percussion transients.",
            "- Relative strengths describe spectral change, not loudness or note count.",
            "- Repeated profiles do not prove the presence of a drum groove.",
            "- This method cannot identify kick, snare, bass, or instruments reliably.",
        ])

        return {
            "text": "\n".join(lines),
            "onset_count": onset_count,
            "onset_rate": onset_rate,
            "bpm": bpm,
            "beat_period_seconds": beat_period,
            "best_phase_seconds": best_phase,
            "phase_score_improvement_percent": phase_improvement,
            "beat_position_strengths": beat_means,
            "eighth_position_strengths": eighth_means,
            "half_beat_strength": halfbeat_mean,
            "bar_profile_deviation": profile_deviation,
            "mean_spectral_flux": mean_flux,
            "median_spectral_flux": median_flux,
        }

    except Exception as exc:
        lines.extend([
            "",
            "Audio onset analysis failed.",
            f"{type(exc).__name__}: {exc}",
        ])
        return {
            "text": "\n".join(lines),
            "error": str(exc),
        }