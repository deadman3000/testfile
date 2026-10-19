
import inspect
import traceback

import torch
import torch.nn.functional as F

from .rhythm_report_helper import analyze_events, analyze_audio_onsets


class YuE2RhythmReferenceDiagnostic:
    """
    Diagnostic node for inspecting the SheetSage2 audio encoder,
    testing reference-audio transcription, and passing ABC through unchanged.

    This node does not modify ABC or generate music.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_encoder": ("AUDIO_ENCODER",),
                "audio": ("AUDIO",),
                "abc": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("abc_passthrough", "analysis_report")
    FUNCTION = "diagnose"
    CATEGORY = "YuE2/Diagnostics"

    @staticmethod
    def _short_repr(value, limit=1500):
        """Convert a result to readable text without flooding the report."""
        try:
            text = repr(value)
        except Exception:
            text = f"<unprintable {type(value).__name__}>"

        if len(text) > limit:
            text = text[:limit] + "... [truncated]"

        return text

    @staticmethod
    def _get_audio_details(audio):
        """
        Extract the standard ComfyUI AUDIO structure:
        {
            'waveform': tensor [batch, channels, samples],
            'sample_rate': int
        }
        """
        if not isinstance(audio, dict):
            raise TypeError(
                "Expected a ComfyUI AUDIO dictionary, "
                f"received {type(audio).__name__}."
            )

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")

        if waveform is None:
            raise ValueError("The AUDIO input has no 'waveform'.")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(
                "The audio waveform is not a PyTorch tensor."
            )

        if waveform.numel() == 0:
            raise ValueError("The audio waveform is empty.")

        if sample_rate is None:
            raise ValueError("The AUDIO input has no sample rate.")

        return waveform, int(sample_rate)

    @staticmethod
    def _prepare_waveform(waveform, original_rate, target_rate):
        """
        Convert ComfyUI audio to mono, resample if required, and return
        a one-dimensional waveform. Batch/channel dimensions are added
        separately when testing the model.
        """
        waveform = waveform.detach().to(dtype=torch.float32)

        if waveform.ndim == 3:
            # Standard ComfyUI shape: [batch, channels, samples].
            # Use the first audio item if the batch contains more than one.
            waveform = waveform[0]
        elif waveform.ndim == 2:
            # Common shape: [channels, samples] or [batch, samples].
            # At this point ComfyUI audio is expected to be channel-first.
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform[0]
        elif waveform.ndim != 1:
            raise ValueError(
                "Unsupported waveform dimensions: "
                f"{list(waveform.shape)}"
            )

        # Convert multi-channel audio to mono when still applicable.
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        original_length = waveform.numel()

        if original_rate != target_rate:
            target_length = max(
                1,
                round(original_length * target_rate / original_rate),
            )

            waveform = F.interpolate(
                waveform.reshape(1, 1, -1),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).reshape(-1)

        return waveform, original_length

    @staticmethod
    def _describe_attributes(obj, exclude_private=True, limit=80):
        if obj is None:
            return ["Object is None."]

        try:
            names = dir(obj)
        except Exception as exc:
            return [f"Could not inspect attributes: {exc}"]

        if exclude_private:
            names = [name for name in names if not name.startswith("_")]

        return names[:limit]

    def _run_transcription(self, model, waveform, device, report):
        """
        Try likely audio tensor shapes used by audio models.

        Return the successful event list so the rhythm helper can analyse
        it without running transcription a second time.
        """
        method = getattr(model, "transcribe", None)

        if not callable(method):
            report.append(
                "\nTRANSCRIPTION TEST\n"
                "No callable transcribe() method was found on the model."
            )
            return None

        try:
            signature = inspect.signature(method)
            report.append(
                f"\nTRANSCRIPTION TEST\ntranscribe signature: {signature}"
            )
        except Exception as exc:
            report.append(
                "\nTRANSCRIPTION TEST\n"
                f"Could not inspect transcribe signature: {exc}"
            )

        try:
            device = torch.device(device)
        except Exception:
            device = torch.device("cpu")

        report.append(f"Input device: {device}")
        report.append(
            f"Prepared waveform shape: {list(waveform.shape)}"
        )

        candidates = [
            ("[batch, channel, samples]", waveform.reshape(1, 1, -1)),
            ("[batch, samples]", waveform.reshape(1, -1)),
        ]

        last_error = None

        for shape_name, candidate in candidates:
            try:
                candidate = candidate.to(device)

                report.append(
                    f"Trying transcribe() with shape {shape_name}: "
                    f"{list(candidate.shape)}"
                )

                with torch.inference_mode():
                    result = method(candidate)

                report.append("Transcription call succeeded.")
                report.append(
                    "Result type: " + type(result).__name__
                )
                report.append(
                    "Result preview: " + self._short_repr(result)
                )

                # Return the events rather than discarding them.
                return result

            except (IndexError, ValueError) as exc:
                last_error = exc
                report.append(
                    f"Attempt with {shape_name} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            except Exception as exc:
                last_error = exc
                report.append(
                    f"Attempt with {shape_name} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                # Try the alternate dimensionality only for likely
                # shape/indexing errors. Do not repeatedly run inference
                # after unrelated errors.
                message = str(exc).lower()
                shape_related = (
                    isinstance(exc, (RuntimeError, IndexError, ValueError))
                    and any(
                        word in message
                        for word in (
                            "dimension",
                            "shape",
                            "index",
                            "size",
                            "channel",
                            "expected",
                        )
                    )
                )

                if not shape_related:
                    break

        if last_error is not None:
            report.append(
                "\nTranscription could not be completed with the "
                "tested waveform shapes."
            )
            report.append(
                "Final exception: "
                f"{type(last_error).__name__}: {last_error}"
            )
            report.append(
                "The installed model may require additional preprocessing "
                "or a different method than transcribe()."
            )

        return None

    def diagnose(self, audio_encoder, audio, abc):
        report = [
            "YuE2 RHYTHM REFERENCE DIAGNOSTIC",
            "Diagnostic node; model compatibility is not guaranteed.",
            "The ABC output is passed through unchanged.",
            "This node does not modify ABC or generate music.",
            "",
        ]

        # Inspect reference audio.
        try:
            waveform, sample_rate = self._get_audio_details(audio)
            report.append("AUDIO INSPECTION")
            report.append(f"Sample rate: {sample_rate}")
            report.append(
                f"Original waveform shape: {list(waveform.shape)}"
            )

            if waveform.ndim == 3:
                mono = waveform[0].mean(dim=0)
            elif waveform.ndim == 2:
                mono = waveform.mean(dim=0)
            else:
                mono = waveform

            report.append(f"Mono waveform shape: {list(mono.shape)}")
            report.append(
                f"Duration: {mono.numel() / sample_rate:.2f} seconds"
            )

        except Exception as exc:
            report.append("AUDIO INSPECTION FAILED")
            report.append(f"{type(exc).__name__}: {exc}")
            report.append(traceback.format_exc())
            report.append("")
            report.append("END OF REPORT")
            return (abc, "\n".join(report))

        # Inspect the connected SheetSage2 encoder.
        report.append("")
        report.append("ENCODER INSPECTION")
        report.append(
            f"Encoder class: {type(audio_encoder).__name__}"
        )
        report.append(
            "Encoder attributes: "
            + ", ".join(self._describe_attributes(audio_encoder))
        )

        model = getattr(audio_encoder, "model", None)
        if model is None:
            report.append(
                "No underlying .model attribute was found on the encoder."
            )
            report.append("")
            report.append("END OF REPORT")
            return (abc, "\n".join(report))

        report.append(f"Underlying model class: {type(model).__name__}")
        report.append(
            "Model attributes: "
            + ", ".join(self._describe_attributes(model))
        )

        target_rate = getattr(
            audio_encoder,
            "model_sample_rate",
            None,
        )

        if target_rate is None:
            target_rate = getattr(model, "sample_rate", sample_rate)

        try:
            target_rate = int(target_rate)
        except Exception:
            target_rate = sample_rate

        report.append(f"Model sample rate: {target_rate}")

        try:
            prepared, original_length = self._prepare_waveform(
                waveform,
                sample_rate,
                target_rate,
            )

            if sample_rate != target_rate:
                report.append(
                    f"Resampled waveform: {sample_rate} Hz -> "
                    f"{target_rate} Hz"
                )
                report.append(
                    "Resampling method: PyTorch linear interpolation."
                )

            report.append(f"Prepared samples: {prepared.numel()}")
            report.append(
                f"Prepared duration: "
                f"{prepared.numel() / target_rate:.2f} seconds"
            )

        except Exception as exc:
            report.append("WAVEFORM PREPARATION FAILED")
            report.append(f"{type(exc).__name__}: {exc}")
            report.append(traceback.format_exc())
            report.append("")
            report.append("END OF REPORT")
            return (abc, "\n".join(report))

        # Prefer the encoder's configured load device, if present.
        device = getattr(audio_encoder, "load_device", None)

        if device is None:
            device = getattr(model, "device", "cpu")

        report.append(f"Selected model device: {device}")

        # Report patcher state but do not force-load or unload models.
        patcher = getattr(audio_encoder, "patcher", None)

        if patcher is not None:
            report.append("ComfyUI model patcher detected.")
            report.append(
                "No manual model load/unload was requested by this node."
            )

        # Run transcription once and retain its result.
        events = self._run_transcription(
            model,
            prepared,
            device,
            report,
        )

        # Existing symbolic analysis.
        rhythm_result = None

        if isinstance(events, (list, tuple)):
            try:
                rhythm_result = analyze_events(events)
                report.append("")
                report.append(rhythm_result["text"])

            except Exception as exc:
                report.append("")
                report.append("RHYTHM HELPER FAILED")
                report.append(f"{type(exc).__name__}: {exc}")
                report.append(traceback.format_exc())

        else:
            report.append("")
            report.append(
                "Rhythm analysis skipped: "
                "transcription returned no event list."
            )

        # New independent audio onset analysis.
        # Uses the original waveform, not the resampled model input.
        try:
            estimated_bpm = (
                rhythm_result.get("median_bpm")
                if isinstance(rhythm_result, dict)
                else None
            )

            audio_result = analyze_audio_onsets(
                mono,
                sample_rate,
                bpm=estimated_bpm,
                beat_offset=0.0,
            )

            report.append("")
            report.append(audio_result["text"])

        except Exception as exc:
            report.append("")
            report.append("AUDIO ONSET ANALYSIS FAILED")
            report.append(f"{type(exc).__name__}: {exc}")
            report.append(traceback.format_exc())

        report.append("")
        report.append("END OF REPORT")

        # Pass the original ABC string through unchanged.
        return (abc, "\n".join(report))


NODE_CLASS_MAPPINGS = {
    "YuE2RhythmReferenceDiagnostic": YuE2RhythmReferenceDiagnostic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2RhythmReferenceDiagnostic": "YuE2 Rhythm Reference Diagnostic",
}