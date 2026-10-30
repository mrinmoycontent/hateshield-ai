HateShield AI
An experimental AI-powered speech and text analysis application built with Python and Streamlit.

HateShield AI analyses uploaded audio and text to demonstrate speech transcription, hate-speech screening, emotion classification, and speaker grouping.
Features
- Audio-to-text transcription using Faster-Whisper
- Automatic speech-language detection
- Three-class text screening: Hate Speech, Offensive Language, and Neither
- Experimental emotion probability analysis
- Estimated speaker grouping and speaker-labelled transcripts
- Transcript and analysis report exports
Technology
- Python
- Streamlit
- Faster-Whisper
- SpeechBrain
- Transformers
- scikit-learn
- TF-IDF and Logistic Regression
- Pandas and NumPy
- Matplotlib
- ReportLab
Application Screenshot
The screenshot below shows the emotion-analysis probability table and chart from a test run.

Emotion scores are model outputs, not validated psychological measurements.
Sample Test Results
The following results were recorded during one application test.
Audio transcription
Metric	Observed result
Detected language	French (fr)
Whisper language probability	57.7%
Whisper segment score	71.6%


The segment score is not transcription accuracy.
Hate-speech screening
Metric	Observed result
Final screening result	Neither
Original model prediction	Neither


Emotion analysis
Emotion	Model probability
Happy	92.48%
Neutral	6.50%
Sad	0.67%
Angry	0.31%
Fear	0.04%


Speaker grouping
- Speaker grouping completed.
- One speaker label was estimated.
- A speaker-labelled transcript was generated.
Speaker labels are estimates and should be checked against the original audio.
Model Information
The text classifier was trained using the Davidson hate-speech and offensive-language dataset. It uses TF-IDF features and Logistic Regression to classify text into three categories:
- 0 — Hate Speech
- 1 — Offensive Language
- 2 — Neither
The reported held-out test accuracy from the training run was 87.47%. Performance varied by class; the Hate Speech category had lower precision and recall than the other categories.
Limitations
- The text classifier was trained on an English-language dataset. Its predictions on French or other non-English text should not be treated as reliable.
- Overall accuracy does not guarantee reliable detection of every hate-speech example.
- Context checks are heuristic and do not establish a speaker's intent.
- Speech recognition can make mistakes, particularly with noisy, short, accented, or mixed-language audio.
- Emotion scores are experimental model estimates, not verified emotional states.
- Speaker grouping is an estimate and can be inaccurate with overlapping speech, noise, or similar voices.
- Human review of the original audio and surrounding context is required.
Project Status
HateShield AI is a portfolio and learning prototype demonstrating how multiple AI components can be combined in a speech-analysis workflow. It is not a production moderation, forensic, or decision-making system.

All predictions should be treated as experimental outputs and should not be used as the sole basis for consequential decisions.
