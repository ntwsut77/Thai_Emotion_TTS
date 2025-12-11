# 🇹🇭 Thai Emotion TTS  
### **Emotion-Aware Thai Speech Generation Using a Hybrid ASR–Emotion–TTS Pipeline**  
**Author:** Natthawut Suthongsa  
**Project Type:** Final-Year Project (Speech Processing / NLP)

---

## 1.📘 Overview  
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

##2. 🧠 System Architecture  
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

---

# 3. Functionality Evaluation

## 3.1 Input → Emotion → TTS Pipeline

### **Input Stage**
Users can either:
- Type Thai text  
- Upload/record Thai audio  

If audio is provided, the system prioritizes it.

---

### **ASR Stage (Whisper Large-V3)**
- Uses chunking (`chunk_length_s=30`) to support long recordings  
- Produces a Thai transcript  
- Handles noisy environments moderately well  

---

### **Preprocessing Stage**
- Normalizes Thai text (`normalize()`)
- Removes unnecessary characters
- *Word tokenization is intentionally disabled* to maintain natural TTS prosody

---

### **Emotion Classification Stage**
Hybrid model:
1. **ML-based** — TF-IDF + LinearSVC trained from corpus.parquet  
2. **Rule-based** — keyword override  
   - “เศร้า”, “สูญเสีย” → sad  
   - “ด่วน”, “ล่าสุด” → excited  
   - “โกรธ”, “เดือด” → angry  

Emotion categories:
- news  
- happy  
- sad  
- angry  
- excited  

---

### **TTS Stage (MMS-Thai TTS)**
Emotion → Prosody mapping (speed, noise):
- **news:** neutral  
- **happy:** faster, lighter  
- **sad:** slower, softer  
- **angry:** fast + high noise  
- **excited:** very fast, strong brightness  

Output: **16-bit PCM WAV**

---

## 3.2 Strengths of the System
- Supports both audio and text input  
- Whisper ASR handles Thai speech with high accuracy  
- Emotion classifier more robust via hybrid logic  
- MMS TTS produces stable, clear Thai voice  
- Works entirely inside Colab with no installation  
- Simple Gradio interface for demo and evaluation  

---

## 3.3 Limitations
- Whisper may misinterpret background noise  
- Emotion in TTS is prosody-based, not deep emotional synthesis  
- Keyword rules may fail on slang/metaphorical expressions  
- MMS-TTS provides only one speaker voice  
- TTS may distort if noise_scale is too high  

---

# 4. Usability Evaluation

## 4.1 How to Run (Google Colab)

### **Step-by-step**
1. Open the notebook:  
   *(Insert link here)*  
2. Click **Runtime → Run all**  
3. Wait until Whisper, Emotion Classifier, and MMS TTS load  
4. When Gradio launches, click the public link to open the demo UI  

---

## 4.2 How to Use the Web Interface

### **Text Input**
- Type any Thai sentence  
- Enable or disable Auto Emotion  

### **Audio Input**
- Upload a `.wav` / `.mp3` file  
- Or use the built-in microphone recorder  

### **Emotion Selection**
- Auto mode uses classifier  
- Manual mode overrides with 5 presets  

### **Output Section**
- Generated Audio (playable WAV)  
- Processed Text  
- Detected Emotion  

---

## 4.3 Suggested Screenshots
<img width="1920" height="629" alt="{371BBD85-12B0-413A-9332-C911C0042D15}" src="https://github.com/user-attachments/assets/903efa50-d9ca-4d05-9e9c-8c7aa85dbec0" />


