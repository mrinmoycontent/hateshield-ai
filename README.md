<div align="center">

# 🛡️ HateShield AI

### AI-powered speech and text analysis

An experimental portfolio project that brings together speech transcription, hate-speech screening, emotion analysis, and speaker grouping in one Streamlit application.

**Python · Streamlit · Faster-Whisper · SpeechBrain · scikit-learn · Transformers**

</div>

---

## ✨ What it does

HateShield AI takes audio or text and runs it through a set of AI-assisted analysis components. It is designed as a practical demonstration of how speech and text-processing models can work together in a single application.

| Capability | Description |
|---|---|
| 🎙️ **Speech transcription** | Converts uploaded audio into text using Faster-Whisper. |
| 🌐 **Language detection** | Displays the detected language and Whisper's language-probability estimate. |
| 🛡️ **Text screening** | Classifies text as Hate Speech, Offensive Language, or Neither. |
| 🎭 **Emotion analysis** | Displays experimental emotion probability scores. |
| 👥 **Speaker grouping** | Estimates speaker labels and creates a speaker-labelled transcript. |
| 📄 **Export** | Supports transcript and analysis report downloads. |

## 🖼️ Application preview

### Emotion analysis — actual test output

The screenshot below is from the running application. It shows the emotion probability table and its corresponding chart.

![HateShield AI Emotion Analysis](assets/emotion-analysis.png)
<p align="center"><em>Observed model output from a test run. Scores are not validated psychological measurements.</em></p>

## 📊 Recorded test results

The following values were captured during one test run of the application.

### Audio transcription

| Output | Recorded value |
|---|---|
| Detected language | French (`fr`) |
| Whisper language probability | 57.7% |
| Whisper segment score | 71.6% |

> The segment score is not transcription accuracy. The language probability is an estimate, not a guarantee that the detected language is correct.

### Hate-speech screening

| Output | Recorded value |
|---|---|
| Final screening result | **Neither** |
| Original model prediction | Neither |

### Emotion analysis

| Emotion | Probability |
|---|---:|
| Happy | 92.48% |
| Neutral | 6.50% |
| Sad | 0.67% |
| Angry | 0.31% |
| Fear | 0.04% |

### Speaker grouping

- Speaker grouping completed successfully in this test.
- **1 speaker label** was estimated.
- A speaker-labelled transcript was generated.

Speaker labels are estimates and must be checked against the original recording.

## 🧠 Model and evaluation

The text-screening model was trained using the **Davidson hate-speech and offensive-language dataset**. Its pipeline uses TF-IDF text features and Logistic Regression to predict three classes:

| Label | Class |
|---:|---|
| `0` | Hate Speech |
| `1` | Offensive Language |
| `2` | Neither |

The reported held-out test accuracy was **87.47%**. Performance was not equal across all classes: the Hate Speech class had lower precision and recall than the other classes. Accuracy alone therefore does not describe how reliably the model identifies hate speech.

## 🧰 Built with

- **Python** — application and model integration
- **Streamlit** — interactive web interface
- **Faster-Whisper** — speech transcription and language detection
- **SpeechBrain** — speaker embedding and grouping
- **Transformers** — experimental emotion analysis
- **scikit-learn** — text classification
- **TF-IDF + Logistic Regression** — hate-speech screening pipeline
- **Pandas, NumPy and Matplotlib** — data handling and visualisation
- **ReportLab** — PDF report generation

## ⚠️ Limitations and responsible use

HateShield AI is an experimental **portfolio prototype**, not a production moderation, forensic, or decision-making system.

- The hate-speech classifier was trained on an **English-language dataset**. Its predictions on French or other non-English transcripts should not be treated as reliable.
- The classifier can make false-positive and false-negative predictions. Its Hate Speech class performed less strongly than the other classes in the reported evaluation.
- Context checks are heuristic; they do not reliably establish a speaker's intent or distinguish every instance of quoting, reporting, or condemning hateful language.
- Speech transcription and language detection can be wrong, particularly with short, noisy, accented, or mixed-language recordings.
- Emotion scores are experimental model outputs, not verified emotional states or psychological assessments.
- Speaker grouping is an estimate and may be inaccurate with overlapping speech, background noise, or similar-sounding voices.
- Always review the original audio, transcript, and surrounding context. Do not use this application as the sole basis for consequential decisions about people.

## 🚀 Project status

The core workflow has been exercised with a test recording: transcription, screening output, emotion probabilities, and speaker grouping were produced. The project is presented as a working learning and portfolio demonstration; the results above are sample outputs, not a claim of production-level accuracy.

---

<div align="center">

**HateShield AI** · Built as an AI/ML portfolio project

</div>
