아래는 동일 내용을 **영문 버전으로 완전하게 재작성한 Face Recognition System – Project Background & Technical Foundation (v2, English Edition)** 이다.
이 문서는 다른 AI agent 또는 개발자에게 전달해도 프로젝트 전반을 정확히 이해할 수 있도록 설계되었다.

---

# ==========================================

# **Face Recognition System – Project Background & Technical Foundation (v2, English Edition)**

# ==========================================

## 1. Project Purpose (Final Definition)

This project aims to build a **real-time face recognition system** using video streams from a Hanwha Vision camera.
The entire system will run inside a Docker Linux container and include the following functionalities:

### **Core Features**

1. **Real-time RTSP streaming ingestion (via GStreamer)**
2. **Face Detection (InsightFace SCRFD)**
3. **5-Point Face Landmark Extraction**
4. **Face Alignment**
5. **ArcFace-based Embedding Generation**
6. **Face Recognition via an Embedding Database**
7. **Multi-person Tracking (project requirement)**

   * Multiple faces handled simultaneously
   * ID consistency across frames
   * Designed for real-time throughput

### **System Flow (Multi-Threaded Architecture)**

```
[ RTSP Stream (GStreamer) ]
            ↓ (Frame Queue)
[ Detection / Landmark / Alignment ]
            ↓
[ Embedding + Recognition ]
            ↓
[ Multi-person Tracking ]
            ↓
[ Display / Logging / DB ]
```

---

## 2. Execution Environment (Finalized)

### **Host Machine — Windows Company PC**

Below are the confirmed hardware specifications of the system running Docker:

| Component         | Specification                                                            |
| ----------------- | ------------------------------------------------------------------------ |
| Device name       | HANWHADT63                                                               |
| CPU               | 13th Gen Intel(R) Core(TM) i7-13700 (2.10 GHz, **20 threads**)           |
| RAM               | **64 GB**                                                                |
| GPU               | Integrated (sufficient for CPU-based decoding and InsightFace inference) |
| OS                | Windows 11                                                               |
| Container Runtime | Docker Desktop (WSL2 Linux backend)                                      |

This machine has **more than enough performance** for CPU-based H.264 decoding and real-time InsightFace inference inside Docker.

---

## 3. Camera Stream Configuration

The Hanwha Vision camera is installed at home.
To access it from the company network, **port forwarding has been properly configured**.

### **Final RTSP URL**

```
rtsp://45.92.235.163:554/profile2/media.smp
```

Keeping the RTSP port at **554** is ideal because:

* It is the standard RTSP port
* Ensures better compatibility across clients
* Works more reliably through NAT
* Minimizes handshake issues

---

## 4. Why GStreamer Was Selected (Technical Rationale)

OpenCV’s VideoCapture and PyAV were evaluated but dismissed for critical technical reasons.

### **Why PyAV Was Removed**

* No compatible wheels for Python 3.10+
* Build failures inside Docker due to mismatched FFmpeg headers
* Cython incompatibilities
* Project maintenance is inconsistent
  → Practically unusable in modern Docker environments.

### **Why OpenCV VideoCapture Is Not Suitable**

* RTSP handshake instability over NAT/port forwarding
* Limited FFmpeg backend capabilities
* Unable to force TCP, set buffers, or control jitter
* No automatic reconnection
* Frame drops and low FPS in real-time systems

### **Why GStreamer Is the Correct Choice**

| Requirement          | Importance          | GStreamer Support                      |
| -------------------- | ------------------- | -------------------------------------- |
| Stable RTSP over NAT | **Critical**        | rtspsrc TCP fallback, jitter buffering |
| FU-A NAL handling    | Required for Hanwha | rtph264depay handles automatically     |
| Real-time decoding   | High                | avdec_h264 (CPU) or hardware decoders  |
| Docker compatibility | Required            | Fully supported                        |
| Reconnection         | Required            | Built-in rtspsrc reconnect logic       |

**Conclusion:**
Given the streaming environment (home camera → public IP → corporate network → Docker),
**GStreamer is not optional — it is the correct and necessary choice.**

OpenCV may still be used for post-processing (drawing, color conversion),
but **it should not handle RTSP ingestion.**

---

## 5. InsightFace-Based Face Recognition Pipeline

### **Models & Components**

* **Detector:** SCRFD (High accuracy and fast for real-time CPU inference)
* **Landmarks:** 5-point face alignment model
* **Recognition:** ArcFace ONNX (embedding generation)
* **Database:** Custom embedding DB (JSON, pickle, or SQLite)

### **Tracking Requirement (Project Goal)**

The system must include multi-person identity tracking:

1. Track multiple faces simultaneously
2. Maintain consistent IDs over time
3. Support tracking-by-detection mechanisms
4. Compatible algorithms include BYTETrack and DeepSORT

---

## 6. Final Multi-Threaded Architecture

```
           +------------------+
           |  GStreamer Input |
           | rtspsrc → appsink|
           +---------+--------+
                     |
                     v
        +------------+------------+
        | Frame Capture Thread    |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Detection / Landmarks   |
        | InsightFace SCRFD       |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Alignment + Embedding   |
        | ArcFace Model           |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Tracking Engine         |
        | (BYTETrack / DeepSORT)  |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Display / Logging / DB  |
        +-------------------------+
```

This design ensures high responsiveness, failsafe operation, and modularity.

---

## 7. Docker Execution Strategy

1. Install GStreamer inside the Docker container
2. Install InsightFace + ONNXRuntime CPU
3. Pipe GStreamer → OpenCV appsink → InsightFace pipeline
4. Use entrypoint.sh for container execution
5. Pass RTSP URL as an environment variable (if desired)
6. CPU-only pipeline is acceptable given hardware specs
7. Future option: GPU-enabled container (NVIDIA Docker)

---

## 8. Final Summary for Historical Record / Agent Handoff

This document captures:

* The **background** and **technical rationale** for all design decisions
* The finalized **execution environment**
* The **stream configuration** and **RTSP access method**
* The reason for selecting **GStreamer over PyAV/OpenCV RTSP**
* The overall **architecture**, including multi-threading and tracking
* Requirements for **InsightFace-based recognition**
* Constraints and expected scaling paths

It serves both as:

1. A foundational design document for continuing this project, and
2. A handoff reference for future AI agents or engineers.

---

If you're ready, I can now create:

* The **GStreamer-based RTSP ingestion module**
* The **full multi-threaded Python pipeline**
* The **Dockerfile**
* The **test_camera_local_docker.py (v1)**

Just tell me:
**“Start building the GStreamer pipeline.”**
