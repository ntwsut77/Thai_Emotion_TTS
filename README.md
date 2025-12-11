# 🇹🇭 Thai Emotion TTS  
### **Emotion-Aware Thai Speech Generation Using a Hybrid ASR–Emotion–TTS Pipeline**  
**Author:** Natthawut Suthongsa  
**Project Type:** Final-Year Project (Speech Processing / NLP)

---

## 📘 Overview  
Emotion-aware speech generation improves clarity, engagement, and human–computer interaction.  
However, **Thai language TTS rarely supports emotion**, mainly due to:

- Lack of emotion-labeled Thai datasets  
- Complex Thai phonology and tonal structure  
- Limited open-source Thai prosody models  
- Fragmented tools for ASR, NLP, and TTS  

This project presents **Thai Emotion TTS**, an integrated pipeline for **Thai news-style speech synthesis with emotional expressiveness**, combining:

- Whisper Large-V3 (ASR)  
- PyThaiNLP text normalization  
- Hybrid Emotion Classifier (TF-IDF + SVM + rule-based override)  
- Facebook MMS (VITS-based Thai TTS)  
- Gradio interface for demonstration  

The goal is to build a **working end-to-end prototype** that can process Thai text or speech input, predict emotion, and output expressive Thai speech.

---

## 🧠 System Architecture  
      ┌────────────────────┐
      │  Text Input (TH)   │
      └─────────┬──────────┘
                │
       OR       │
                ▼
      ┌────────────────────┐
      │  Audio Input (TH)  │
      └─────────┬──────────┘
                │  (librosa / Whisper)
                ▼
      ┌────────────────────┐
      │  Whisper ASR       │  → Thai Transcript
      └─────────┬──────────┘
                ▼
      ┌────────────────────┐
      │  PyThaiNLP         │  → cleaned text
      └─────────┬──────────┘
                ▼
      ┌───────────────────────────────────────┐
      │  Emotion Module                       │
      │  - TF-IDF + LinearSVC classifier      │
      │  - Rule-based keyword override        │
      └─────────┬────────────────────────────┘
                ▼
      ┌───────────────────────────────────────┐
      │ MMS: facebook/mms-tts-tha (VITS TTS)  │
      │ - Thai speech synthesis               │
      │ - Conceptual prosody adjustment       │
      └─────────┬────────────────────────────┘
                ▼
      ┌────────────────────┐
      │   Audio Output     │
      └────────────────────┘

---

## 🎯 Background & Motivation  
Emotion plays a crucial role in **news narration**, **digital storytelling**, **audiobooks**, and **AI voice agents**.  
While English emotion-TTS is widely supported, **Thai TTS remains monotonic**, with major gaps:

- No robust open-source Thai SER datasets  
- Prosody control is underdeveloped  
- Existing Thai TTS models lack emotional conditioning  
- Thai ASR historically lags behind English benchmarks  

This project aims to fill these practical gaps, focusing on:

1. **End-to-end integration** of Thai ASR → Emotion → TTS  
2. **Practicality** within limited GPU/CPU resources (suitable for student research)  
3. **Explainability** through hybrid rule-based + ML emotion detection  
4. **Feasibility** using open models with Thai support (Whisper, MMS TTS)

---

## ⚙️ Methodology  

### **1. Automatic Speech Recognition (Whisper-large-v3)**  
- Whisper is used to convert optional speech input into text.  
- `chunk_length_s = 30` ensures long audio is supported without memory overflow.  
- Works well on Thai due to training on multilingual datasets.  

### **2. Text Preprocessing (PyThaiNLP)**  
- Uses `normalize()` to clean Thai text, remove weird Unicode, and unify tone marks.  
- Intentional choice: **no tokenization** → keeps phrasing natural for TTS prosody.  
- Suitable for news-style delivery where pause boundaries matter.

---

### **3. Emotion Classification (Hybrid Model)**  
Emotion categories:  
`news`, `happy`, `sad`, `angry`, `excited`

#### **Machine Learning Component**
- TF-IDF vectorizer (20k vocabulary)  
- LinearSVC classifier  
- Trained on news-like content loaded from `corpus.parquet`  
- Balanced performance and fast inference  

#### **Rule-Based Component**  
Handles explicit emotional cues via Thai keywords  
(เช่น “เศร้า”, “ดีใจ”, “เดือด”)  
This improves accuracy in sentiment-heavy sentences.

---

### **4. Thai Text-to-Speech (facebook/mms-tts-tha)**  
The model is based on **VITS: Variational Inference Text-to-Speech**, supporting:

- End-to-end phoneme-to-waveform generation  
- Smooth, natural Thai pronunciation  
- Fast waveform generation  

**Prosody Control (Conceptual Mapping):**  
Emotion → `(speed, noise)` parameters conceptually mapped to indicate expressive changes.  
Although MMS does not expose built-in emotional embeddings, conceptual prosody adjustment provides audible variation.

---

### **5. Output Module (Gradio)**  
Gradio is used to deliver:

- Audio output (.wav)  
- Clean text  
- Detected emotion  
- Interactive sharing (public demo link)

---

## 📊 Results & Analysis  
### **1. ASR Performance**
Whisper-large-v3 produced reliable Thai transcripts even with varying accents and background noise.

### **2. Emotion Classification**
- TF-IDF + SVM baseline: stable performance on structured text  
- Rule-based override correctly fixes ML mispredictions  
- Effective for news-style emotional cues

### **3. TTS Emotional Expression**
MMS Thai TTS generated clear pronunciation with conceptual variations such as:

- **Happy** → lighter, slightly faster  
- **Sad** → slower, lower tone  
- **Angry** → sharper onset, slightly increased noise  
- **Excited** → higher energy  

While not true emotional embeddings, the conceptual prosody mapping enhances perceived expression.

---

## ⚠️ Limitations  
- MMS lacks built-in emotional conditioning → limits expressiveness  
- Rule-based emotion classification fails on subtle context  
- Long sentences reduce prosody accuracy  
- Multi-speaker TTS not supported  
- Emotion-to-prosody mapping remains conceptual, not learned

---

## 🧭 Future Work  
- Train Thai emotion embeddings via VITS fine-tuning  
- Extend corpus with labeled emotional audio  
- Integrate Multi-speaker TTS models  
- Introduce objective metrics (MOS, SER accuracy alignment)  
- Mobile/Realtime deployment via ONNX or quantized inference  

---


