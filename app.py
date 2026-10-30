import os
import io
import math
import tempfile
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from xml.sax.saxutils import escape


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="HateShield AI | Speech & Text Analysis",
    page_icon="🛡️",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(APP_DIR / "hate_speech_model.pkl")
LABEL_PATH = str(APP_DIR / "label_mapping.pkl")

st.title("🛡️ HateShield AI")
st.caption(
    "Multilingual speech transcription, experimental emotion analysis, "
    "and hate-speech screening."
)

st.info(
    "Research prototype: model predictions can be wrong. "
    "Review context and avoid treating automated results as final judgments."
)


# ---------------------------------------------------------
# EXISTING HATE-SPEECH MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_hate_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)

    labels = None
    if os.path.exists(LABEL_PATH):
        try:
            labels = joblib.load(LABEL_PATH)
        except Exception:
            labels = None

    return model, labels


def get_label(class_id, labels):
    """Support common label-map formats and model class labels."""
    if labels is not None:
        try:
            if isinstance(labels, dict):
                if class_id in labels:
                    return str(labels[class_id])
                if str(class_id) in labels:
                    return str(labels[str(class_id)])

                # Also support reverse mappings such as
                # {"Hate Speech": 0, "Offensive Language": 1}.
                for key, value in labels.items():
                    if value == class_id or str(value) == str(class_id):
                        return str(key)

            if isinstance(labels, (list, tuple, np.ndarray)):
                return str(labels[int(class_id)])
        except Exception:
            pass

    defaults = {
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neither",
    }
    try:
        return defaults.get(int(class_id), str(class_id))
    except Exception:
        return str(class_id)


def predict_text(text, model, labels):
    """Run the existing model and return labels and available probabilities."""
    prediction = model.predict([text])[0]
    predicted_label = get_label(prediction, labels)

    probabilities = {}
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba([text])[0]
        classes = getattr(model, "classes_", range(len(raw_probabilities)))

        for class_id, probability in zip(classes, raw_probabilities):
            probabilities[get_label(class_id, labels)] = float(probability)

    return predicted_label, probabilities


def apply_context_checks(text, content_type, model_label):
    """
    Conservative context checks. These rules are not a substitute
    for a human review and do not establish intent.
    """
    lower = text.lower()
    final_label = model_label
    reasons = []

    group_pattern = (
        r"\b(people|members|community|communities|group|"
        r"immigrants|refugees|women|men|religious people|"
        r"ethnic group|race|racial group)\b"
    )
    violence_pattern = (
        r"\b(should be eliminated|must be eliminated|"
        r"kill|eliminate|exterminate|eradicate|"
        r"wipe out|murder|slaughter|destroy)\b"
    )
    reporting_pattern = (
        r"\b(reported|reportedly|police said|officials said|"
        r"according to|the article says|the report says|"
        r"investigating|alleged|allegedly|quoted as saying)\b"
    )

    group_mentioned = __import__("re").search(group_pattern, lower)
    violent_language = __import__("re").search(violence_pattern, lower)
    reporting_context = __import__("re").search(reporting_pattern, lower)

    if group_mentioned and violent_language:
        if reporting_context:
            final_label = "Needs human review"
            reasons.append(
                "The text includes reporting-style context and group-directed "
                "violent language. Check whether it is quoting or describing "
                "a threat rather than endorsing it."
            )
        elif content_type == "Social media post/comment":
            final_label = "Hate Speech"
            reasons.append(
                "A rule-based check detected potentially group-directed "
                "violent language. Human review is recommended."
            )

    # Avoid interpreting ordinary product criticism as hate speech
    # when the text clearly refers to a product or service.
    product_pattern = (
        r"\b(product|service|app|application|website|phone|"
        r"laptop|movie|film|company|brand|software)\b"
    )
    criticism_pattern = r"\b(hate|stupid|terrible|awful|worst)\b"

    if (
        __import__("re").search(product_pattern, lower)
        and __import__("re").search(criticism_pattern, lower)
        and model_label == "Hate Speech"
    ):
        final_label = "Offensive Language"
        reasons.append(
            "The text appears to criticise a product or service. "
            "The automated label was adjusted; verify the context."
        )

    if content_type == "News article" and not reasons:
        reasons.append(
            "News requires context review, particularly for quotations, "
            "reported threats, and descriptions of events."
        )

    return final_label, reasons


def screen_text(text, content_type, model, labels):
    model_label, probabilities = predict_text(text, model, labels)
    final_label, reasons = apply_context_checks(
        text, content_type, model_label
    )
    return {
        "text": text,
        "content_type": content_type,
        "model_label": model_label,
        "final_label": final_label,
        "probabilities": probabilities,
        "reasons": reasons,
    }


def display_screening(result):
    st.subheader("Hate-speech screening")

    st.metric("Final screening result", result["final_label"])
    st.caption(f"Original model prediction: {result['model_label']}")

    if result["probabilities"]:
        df = pd.DataFrame({
            "Class": list(result["probabilities"].keys()),
            "Probability": [
                round(v * 100, 2)
                for v in result["probabilities"].values()
            ],
        }).sort_values("Probability", ascending=False)

        st.write("Original model probabilities")
        st.bar_chart(df.set_index("Class"))

    for reason in result["reasons"]:
        st.warning(reason)

    if not result["reasons"]:
        st.caption(
            "No additional context rule was triggered. "
            "This does not guarantee that the prediction is correct."
        )


# ---------------------------------------------------------
# MULTILINGUAL SPEECH TRANSCRIPTION
# ---------------------------------------------------------
@st.cache_resource
def load_whisper():
    # Compatibility shim: newer PyAV releases removed metadata_errors,
    # while some Faster-Whisper releases still pass it to av.open().
    import av
    if not getattr(av.open, "_hateshield_compat", False):
        _original_av_open = av.open

        def _compatible_av_open(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return _original_av_open(*args, **kwargs)

        _compatible_av_open._hateshield_compat = True
        av.open = _compatible_av_open

    from faster_whisper import WhisperModel

    # "base" balances download size and transcription speed.
    # CPU/int8 works without an API key or GPU.
    return WhisperModel("small", device="cpu", compute_type="int8")



@st.cache_resource
def load_speaker_encoder():
    """Load an ungated pretrained speaker embedding model; no HF token field required."""
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / ".cache" / "hateshield" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )


def diarize_audio(audio_bytes, filename, transcript_segments):
    """Estimate anonymous speakers from overlapping speech windows.

    The embedding windows are independent of Whisper sentence boundaries, so
    short transcript segments do not automatically disappear from diarization.
    Returns time-stamped speaker windows for mapping onto the transcript.
    """
    if not transcript_segments:
        raise ValueError("Transcribe the audio first; no timestamped segments were found.")

    import torch
    import torchaudio
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    suffix = Path(filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a"}:
        raise ValueError("Upload a WAV, MP3, or M4A audio file.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            temp_path = f.name

        waveform, sample_rate = torchaudio.load(temp_path)
        waveform = waveform.mean(dim=0).float()
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        total_samples = waveform.numel()
        duration = total_samples / 16000.0
        if duration < 1.5:
            raise ValueError("Audio is too short for speaker grouping. Use a longer recording.")

        # Limit embedding windows to regions Whisper marked as speech, while
        # retaining enough surrounding audio for a useful voice embedding.
        speech_spans = []
        for seg in transcript_segments:
            a = max(0.0, float(seg.get("start", 0.0)))
            b = min(duration, float(seg.get("end", 0.0)))
            if b > a:
                speech_spans.append((a, b))
        if not speech_spans:
            raise ValueError("No valid timestamped speech segments were found.")

        encoder = load_speaker_encoder()
        window_seconds = min(3.0, duration)
        step_seconds = 0.75
        window_samples = max(int(1.5 * 16000), int(window_seconds * 16000))
        step_samples = int(step_seconds * 16000)
        final_start = max(0, total_samples - window_samples)
        starts = list(range(0, final_start + 1, step_samples))
        if not starts or starts[-1] != final_start:
            starts.append(final_start)

        embeddings, windows = [], []
        for start_sample in sorted(set(starts)):
            end_sample = min(total_samples, start_sample + window_samples)
            start_sec, end_sec = start_sample / 16000.0, end_sample / 16000.0
            speech_overlap = sum(max(0.0, min(end_sec, b) - max(start_sec, a))
                                 for a, b in speech_spans)
            if speech_overlap < 0.35:
                continue
            chunk = waveform[start_sample:end_sample]
            # ECAPA works best with a little audio; pad only if the final chunk is short.
            if chunk.numel() < int(1.5 * 16000):
                chunk = torch.nn.functional.pad(chunk, (0, int(1.5 * 16000) - chunk.numel()))
            with torch.no_grad():
                emb = encoder.encode_batch(chunk.unsqueeze(0)).squeeze().detach().cpu().numpy()
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if np.all(np.isfinite(emb)) and np.linalg.norm(emb) > 0:
                embeddings.append(emb)
                windows.append({"start": start_sec, "end": end_sec, "speaker": "Speaker unknown"})

        if not embeddings:
            raise RuntimeError("No usable speech embeddings were produced. Try a clearer, longer recording.")

        vectors = normalize(np.asarray(embeddings))
        if len(vectors) == 1:
            cluster_labels = np.zeros(1, dtype=int)
        else:
            # A moderate cosine-distance threshold is less likely to split one
            # voice into many labels because of changes in tone or loudness.
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=0.40,
                    metric="cosine", linkage="average"
                )
            except TypeError:  # compatibility with older scikit-learn
                clusterer = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=0.40,
                    affinity="cosine", linkage="average"
                )
            cluster_labels = clusterer.fit_predict(vectors)

        # Name speakers by their first appearance in the recording.
        names = {}
        for window, label in zip(windows, cluster_labels):
            label = int(label)
            if label not in names:
                names[label] = f"Speaker {len(names) + 1}"
            window["speaker"] = names[label]
        return windows
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def attach_speakers_to_segments(segments, turns):
    """Assign each transcript segment using overlap-weighted speaker votes."""
    labelled = []
    for segment in segments:
        votes = {}
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        for turn in turns:
            overlap = max(
                0.0,
                min(seg_end, float(turn.get("end", 0.0)))
                - max(seg_start, float(turn.get("start", 0.0))),
            )
            if overlap > 0 and turn.get("speaker") != "Speaker unknown":
                votes[turn["speaker"]] = votes.get(turn["speaker"], 0.0) + overlap
        item = dict(segment)
        item["speaker"] = max(votes, key=votes.get) if votes else "Speaker unknown"
        labelled.append(item)
    return labelled


def transcribe_audio(audio_bytes, filename):
    """Transcribe in the detected language, with a quality-checked English retry
    when Whisper's automatic language detection is uncertain.

    The English retry is only selected when its segment-quality score is better;
    it does not blindly force English and overwrite genuine non-English speech.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a"}:
        raise ValueError("Upload a WAV, MP3, or M4A audio file.")

    temporary_path = None

    def run_pass(model, path, language=None):
        segments, info = model.transcribe(
            path,
            language=language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        rows = []
        quality_values = []
        confidence_values = []
        for seg in segments:
            phrase = (getattr(seg, "text", "") or "").strip()
            if not phrase:
                continue
            rows.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": phrase,
            })
            avg_logprob = getattr(seg, "avg_logprob", None)
            if avg_logprob is not None:
                avg_logprob = float(avg_logprob)
                compression = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
                no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                # Ranking heuristic only; it is not a calibrated accuracy score.
                quality_values.append(
                    avg_logprob
                    - max(0.0, compression - 2.4) * 0.5
                    - max(0.0, no_speech - 0.5) * 0.5
                )
                confidence_values.append(math.exp(min(0.0, avg_logprob)))
        quality = sum(quality_values) / len(quality_values) if quality_values else -999.0
        confidence = (sum(confidence_values) / len(confidence_values) * 100) if confidence_values else None
        return {
            "rows": rows,
            "info": info,
            "quality": quality,
            "confidence": confidence,
        }

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temporary_path = temp_file.name

        whisper = load_whisper()
        best = run_pass(whisper, temporary_path, language=None)
        detected_language = getattr(best["info"], "language", "unknown")
        language_probability = getattr(best["info"], "language_probability", None)

        # The previous bug: Whisper could label clearly English speech as French
        # and produce phonetic-looking text. If language confidence is low, compare
        # an English pass instead of trusting the first language guess blindly.
        try_english = (
            detected_language != "en"
            and language_probability is not None
            and float(language_probability) < 0.80
        )
        if try_english:
            english = run_pass(whisper, temporary_path, language="en")
            if english["rows"] and (
                not best["rows"] or english["quality"] > best["quality"] + 0.05
            ):
                best = english
                detected_language = "en"
                # Forced-language pass does not provide a meaningful automatic
                # language probability; show None rather than the French guess.
                language_probability = None

        collected = best["rows"]
        transcript = "\n".join(item["text"] for item in collected).strip()
        confidence_estimate = best["confidence"]

        # Keep original-language transcript unchanged. Translation is used only
        # as a separate input to the English-focused experimental emotion model.
        emotion_text = transcript
        emotion_translation_note = None
        if transcript and detected_language != "en":
            try:
                translated_segments, _ = whisper.transcribe(
                    temporary_path,
                    language=detected_language if detected_language != "unknown" else None,
                    task="translate",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                translated_text = " ".join(
                    part.text.strip() for part in translated_segments
                    if part.text and part.text.strip()
                ).strip()
                if translated_text:
                    emotion_text = translated_text
                    emotion_translation_note = (
                        "Emotion estimates use Whisper's automatic English translation; "
                        "translation may change nuance and the result is experimental."
                    )
                else:
                    emotion_text = ""
                    emotion_translation_note = "Automatic English translation returned no text; emotion prediction is unavailable."
            except Exception as translation_error:
                emotion_text = ""
                emotion_translation_note = (
                    "Automatic English translation for emotion analysis failed: "
                    f"{type(translation_error).__name__}: {translation_error}"
                )

        return {
            "transcript": transcript,
            "emotion_text": emotion_text,
            "emotion_translation_note": emotion_translation_note,
            "segments": collected,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "confidence_estimate": confidence_estimate,
        }
    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


# ---------------------------------------------------------
# EXPERIMENTAL EMOTION CLASSIFICATION
# ---------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=-1,
    )


def predict_emotions(text):
    """
    Convert the English model's seven categories into the five
    requested display categories. This is a simplified mapping.
    """
    classifier = load_emotion_model()
    raw = classifier(text[:4000])

    # Some Transformers versions return a nested list for a single input.
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    source = {
        item["label"].strip().lower(): float(item["score"])
        for item in raw
    }

    five = {
        "Happy": source.get("joy", 0.0),
        "Sad": source.get("sadness", 0.0),
        "Angry": (
            source.get("anger", 0.0)
            + source.get("disgust", 0.0)
        ),
        "Fear": source.get("fear", 0.0),
        "Neutral": (
            source.get("neutral", 0.0)
            + source.get("surprise", 0.0)
        ),
    }

    total = sum(five.values())
    if total > 0:
        five = {key: value / total for key, value in five.items()}

    return five


def make_pdf(title, content):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(escape(title), styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "Generated by HateShield AI — research prototype; "
            "automated outputs require human review.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    for line in content.splitlines():
        safe_line = escape(line) if line.strip() else " "
        story.append(Paragraph(safe_line, styles["BodyText"]))
        story.append(Spacer(1, 5))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
model, label_mapping = load_hate_model()

if model is None:
    st.error(
        "Existing hate-speech model not found at "
        "/content/hate_speech_model.pkl. "
        "Restore your trained model file before using text screening."
    )

text_tab, audio_tab, about_tab = st.tabs([
    "📝 Text screening",
    "🔊 Audio transcription",
    "ℹ️ About & limitations",
])


def clear_text_input():
    """Clear the text widget before Streamlit instantiates it on rerun."""
    st.session_state["screening_text"] = ""
    st.session_state.pop("text_result", None)


# ---------------------------------------------------------
# TAB 1: TEXT SCREENING
# ---------------------------------------------------------
with text_tab:
    st.subheader("Screen a post, comment, or news article")

    content_type = st.radio(
        "Content type",
        ["Social media post/comment", "News article"],
        horizontal=True,
        key="text_content_type",
    )

    text_input = st.text_area(
        "Enter text to analyse",
        height=180,
        placeholder="Paste the text here...",
        key="screening_text",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        analyse_text = st.button(
            "Analyse text",
            type="primary",
            use_container_width=True,
        )

    with col_b:
        st.button(
            "Clear text",
            use_container_width=True,
            on_click=clear_text_input,
        )

    if analyse_text:
        if not text_input.strip():
            st.warning("Enter some text before analysing.")
        elif model is None:
            st.error("The hate-speech model file is missing.")
        else:
            with st.spinner("Analysing text..."):
                try:
                    st.session_state["text_result"] = screen_text(
                        text_input.strip(),
                        content_type,
                        model,
                        label_mapping,
                    )
                except Exception as exc:
                    st.error(
                        f"Text analysis failed: {type(exc).__name__}: {exc}"
                    )

    if st.session_state.get("text_result"):
        display_screening(st.session_state["text_result"])


# ---------------------------------------------------------
# TAB 2: AUDIO TRANSCRIPTION + ANALYSIS
# ---------------------------------------------------------
with audio_tab:
    st.subheader("Upload speech audio")

    st.caption("Language detection is automatic. No manual language selection is required.")

    audio_file = st.file_uploader(
        "Upload WAV, MP3, or M4A",
        type=["wav", "mp3", "m4a"],
        key="audio_upload",
    )

    transcribe_button = st.button(
        "Transcribe audio",
        type="primary",
        disabled=(audio_file is None),
        use_container_width=True,
    )

    if transcribe_button and audio_file is not None:
        with st.spinner(
            "Transcribing audio. The first run may download the model..."
        ):
            try:
                audio_result = transcribe_audio(
                    audio_file.getvalue(),
                    audio_file.name,
                )
                st.session_state["audio_result"] = audio_result
                st.session_state["audio_bytes_for_diarization"] = audio_file.getvalue()
                st.session_state["audio_filename_for_diarization"] = audio_file.name
                st.session_state.pop("diarized_segments", None)
                st.session_state.pop("speaker_turns", None)
                st.session_state.pop("audio_screening_result", None)
                st.session_state.pop("emotion_result", None)
            except Exception as exc:
                st.error(
                    f"Transcription failed: {type(exc).__name__}: {exc}"
                )

    audio_result = st.session_state.get("audio_result")

    if audio_result:
        st.markdown("### Transcript")

        detected_language = audio_result["detected_language"]
        st.write(f"**Detected language code:** `{detected_language}`")

        lang_prob = audio_result["language_probability"]
        if lang_prob is not None:
            st.write(
                f"**Whisper language probability (estimate):** {lang_prob * 100:.1f}%"
            )
            if lang_prob < 0.75:
                st.warning(
                    "Whisper is uncertain about the language in this audio. "
                    "The transcript was still generated automatically; no manual "
                    "language selection is required. Review the transcript against "
                    "the audio, especially if the clip is short, noisy, or contains music."
                )

        transcript = audio_result["transcript"]

        st.text_area(
            "Recognised speech",
            value=transcript,
            height=220,
            key="transcript_display",
        )

        confidence = audio_result["confidence_estimate"]
        if confidence is not None:
            st.metric(
                "Whisper segment score (not accuracy)",
                f"{confidence:.1f}%",
            )
            st.caption(
                "This is a rough average derived from Whisper's segment "
                "log-probabilities. It is NOT transcription accuracy, a "
                "validated confidence percentage, or a guarantee that the "
                "recognised words are correct. Check the transcript against the audio."
            )
        else:
            st.caption("A transcription score was not available for this audio.")

        # TXT download
        txt_content = (
            "HateShield AI — Audio Transcript\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Detected language: {detected_language}\n\n"
            f"{transcript}\n"
        )

        # PDF download
        pdf_content = (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Detected language: {detected_language}\n\n"
            f"Transcript:\n{transcript}\n\n"
            "Note: Automated transcription can contain errors. "
            "Please review before relying on it."
        )

        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                "⬇️ Download transcript (.txt)",
                data=txt_content.encode("utf-8"),
                file_name="hateshield_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl2:
            try:
                pdf_bytes = make_pdf(
                    "HateShield AI — Audio Transcript",
                    pdf_content,
                )
                st.download_button(
                    "⬇️ Download transcript (.pdf)",
                    data=pdf_bytes,
                    file_name="hateshield_transcript.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(f"PDF generation failed: {exc}")

        st.divider()
        st.subheader("Analyse the recognised speech")

        if st.button(
            "Run hate-speech screening on transcript",
            disabled=(not transcript or model is None),
            use_container_width=True,
        ):
            with st.spinner("Screening the transcript..."):
                try:
                    st.session_state["audio_screening_result"] = screen_text(
                        transcript,
                        "Social media post/comment",
                        model,
                        label_mapping,
                    )
                except Exception as exc:
                    st.error(f"Screening failed: {exc}")

        if st.session_state.get("audio_screening_result"):
            display_screening(
                st.session_state["audio_screening_result"]
            )

        st.divider()
        st.subheader("Emotion prediction (experimental)")

        st.caption(
            "The emotion model is English-focused. Results for other "
            "languages may be unreliable. These labels are not evidence "
            "of a person's actual feelings, intentions, personality, or suitability."
        )


        detected_code = str(audio_result.get("detected_language", "")).lower()
        emotion_input = audio_result.get("emotion_text", transcript)
        emotion_supported = bool(str(emotion_input).strip())
        translation_note = audio_result.get("emotion_translation_note")

        if detected_code != "en":
            st.info(
                "Language detection and transcription remain automatic. "
                "For this English-focused emotion model, HateShield AI automatically "
                "translates non-English speech to English for the experimental text estimate; "
                "the original transcript below is not changed."
            )
        if translation_note:
            st.caption(translation_note)
        if not emotion_supported:
            st.warning(
                "Emotion prediction is unavailable because an English text input "
                "could not be prepared automatically. Transcription is still available."
            )

        if st.button(
            "Predict emotion probabilities",
            disabled=(not transcript.strip() or not emotion_supported),
            use_container_width=True,
        ):
            with st.spinner("Loading the emotion model if needed..."):
                try:
                    st.session_state["emotion_result"] = predict_emotions(str(emotion_input))
                except Exception as exc:
                    st.error(
                        "Emotion analysis failed. Check the internet "
                        "connection and installed Transformers version. "
                        f"Details: {type(exc).__name__}: {exc}"
                    )

        emotions = st.session_state.get("emotion_result")

        if emotions:
            emotion_df = pd.DataFrame({
                "Emotion": list(emotions.keys()),
                "Probability (%)": [
                    round(value * 100, 2)
                    for value in emotions.values()
                ],
            }).sort_values("Probability (%)", ascending=False)

            st.dataframe(
                emotion_df,
                use_container_width=True,
                hide_index=True,
            )

            chart_df = emotion_df.set_index("Emotion")
            st.bar_chart(chart_df)

            st.caption(
                "The five categories are simplified from the source "
                "model's original labels. Scores are model outputs, not "
                "validated psychological measurements."
            )

    st.divider()
    st.subheader("Speaker separation")
    st.caption(
        "Experimental speaker grouping uses a pretrained speaker-embedding model. "
        "No Hugging Face access token is required. Short turns, overlapping speech, "
        "noise, and similar voices can cause mistakes; labels are anonymous estimates."
    )

    if st.button(
        "Run speaker separation",
        disabled=not bool(audio_result and audio_result.get("segments")),
        use_container_width=True,
        key="run_speaker_diarization",
    ):
        with st.spinner("Loading speaker model and grouping transcript turns..."):
            try:
                saved_audio = st.session_state.get("audio_bytes_for_diarization")
                saved_filename = st.session_state.get("audio_filename_for_diarization")
                if not saved_audio or not saved_filename:
                    raise ValueError("Upload and transcribe the audio first.")
                speaker_turns = diarize_audio(
                    saved_audio, saved_filename, audio_result.get("segments", [])
                )
                diarized_segments = attach_speakers_to_segments(
                    audio_result.get("segments", []), speaker_turns
                )
                st.session_state["speaker_turns"] = speaker_turns
                st.session_state["diarized_segments"] = diarized_segments
                detected = len({
                    t["speaker"] for t in speaker_turns
                    if t["speaker"] != "Speaker unknown"
                })
                st.success(
                    f"Speaker grouping finished: {detected} speaker label(s) estimated. "
                    "Please review against the audio."
                )
            except Exception as exc:
                st.error(
                    "Speaker separation could not run. Check the error below; "
                    "the transcript itself is unchanged. "
                    f"{type(exc).__name__}: {exc}"
                )

    if st.session_state.get("diarized_segments"):
        st.markdown("**Speaker-labelled transcript**")
        diarized_text = "\n".join(
            f"[{item['speaker']}] {item['text']}"
            for item in st.session_state["diarized_segments"]
        )
        st.text_area(
            "Diarized transcript",
            value=diarized_text,
            height=220,
            key="diarized_transcript_display",
        )
        st.download_button(
            "Download speaker-labelled transcript (.txt)",
            data=diarized_text.encode("utf-8"),
            file_name="hateshield_diarized_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ---------------------------------------------------------
# TAB 3: ABOUT
# ---------------------------------------------------------
with about_tab:
    st.subheader("What this prototype does")

    st.markdown("""
    - Transcribes WAV, MP3, and M4A audio with a multilingual Whisper model.
    - Provides TXT and PDF transcript downloads.
    - Shows a rough transcription score when available.
    - Screens text or recognised speech using your existing classifier.
    - Displays experimental emotion scores using English text; non-English speech is
      automatically translated for the emotion estimate while the original transcript is preserved.
    - Offers experimental speaker grouping using speaker embeddings without a Hugging Face token; labels require human review.
    - Supports news context checks and flags selected cases for human review.

    **Important limitations**

    - Transcription and classification can be wrong.
    - The Whisper segment score is not transcription accuracy.
    - Emotion predictions are experimental and English-focused. Non-English speech
      is automatically translated to English for the text-based estimate; translation
      may change nuance and does not measure vocal tone.
    - Speaker grouping is approximate and may mislabel short turns, overlap, noise, or similar voices.
    - Do not use emotion predictions to evaluate job candidates, workers,
      or a person's mental state.
    - Do not use automated hate-speech results as the sole basis for
      moderation, discipline, or other consequential decisions.
    """)
