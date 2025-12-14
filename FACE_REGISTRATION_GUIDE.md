# Face Registration Guide

## Quick Start

To register a new person's face, simply run:

```bash
python register_new_person.py
```

The script will guide you through the entire process!

---

## How It Works

### Step 1: Run the Registration Helper
```bash
python register_new_person.py
```

### Step 2: Enter the Person's Name
The script will ask you to enter the name of the person you want to register.

### Step 3: Position the Person in Front of the Camera
- The person should be facing the camera
- Good lighting helps improve accuracy
- **Multiple people can be in the frame** - the system will ask you to select which face to register

### Step 4: The System Automatically:
1. **Connects to the camera stream**
2. **Captures multiple frames** (5 seconds) to find the best quality image
3. **Detects all faces** in the camera view
4. **Compares faces against the database** to identify:
   - **KNOWN faces** - People already registered (shows their name)
   - **UNKNOWN faces** - People not yet registered

### Step 5: Select the Face to Register
- If **only ONE unknown face** is detected → System automatically registers it
- If **multiple unknown faces** are detected → You choose which one to register
- If **all faces are known** → Registration is cancelled (the person might already be registered)

### Step 6: Confirmation
Once registered, the person's name will immediately appear in the live video stream!

---

## Example Scenarios

### Scenario 1: Solo Registration (Easiest)
**Situation:** Only the new person (e.g., "Alice") is in front of the camera

**What happens:**
```
Detected faces:
  [1] UNKNOWN: Age 25, Female, Score 0.95, Similarity 0.25

Found 1 unknown face - registering as Alice
Successfully registered Alice!
```

### Scenario 2: Multiple People Present
**Situation:** SungHwan (already registered) and YounHo (new person) are both in the camera

**What happens:**
```
Detected faces:
  [1] KNOWN: Age 45, Male, Score 0.92, Similarity 0.78 - Matches: SungHwan
  [2] UNKNOWN: Age 12, Male, Score 0.89, Similarity 0.31

Found 1 unknown face - registering as YounHo
Successfully registered YounHo!
```

### Scenario 3: Multiple New People
**Situation:** Alice and Bob are both new (neither registered yet)

**What happens:**
```
Detected faces:
  [1] UNKNOWN: Age 25, Female, Score 0.95, Similarity 0.25
  [2] UNKNOWN: Age 30, Male, Score 0.91, Similarity 0.28

Found 2 unknown faces.
Which face belongs to Alice?
  [1] Age 25, Female, Score 0.95
  [2] Age 30, Male, Score 0.91

Enter number (1-2), or 0 to cancel: 1

Successfully registered Alice!
```

---

## Improving Recognition Accuracy (Update Existing Faces)

### Why Update Embeddings?

If a registered person's recognition accuracy is low (similarity score < 0.65), you can improve it by adding more face samples from different angles. This increases the similarity score from ~0.50-0.60 to **0.65-0.85** or higher!

### Quick Start: Update Existing Person

To add more face samples for better accuracy:

```bash
python update_person_embeddings.py
```

The script will:
1. Show all registered people and their current embedding counts
2. Ask which person to update
3. Ask how many additional samples to capture (default: 2, recommended)
4. Guide you through capturing different poses

### How the Update Process Works

#### Step 1: Run the Update Helper
```bash
python update_person_embeddings.py
```

#### Step 2: Select Person to Update
The system shows all registered people:
```
People in database:
  - SungHwan (1 embedding(s))
  - JeeYoung (1 embedding(s))

Enter the name of the person to update: SungHwan
```

#### Step 3: Choose Number of Samples
```
Number of additional samples to capture (default: 2): 2
```

**Recommended:** 2 additional samples (total of 3 embeddings)
- More samples = better accuracy
- Diminishing returns after 3-4 samples

#### Step 4: Follow Guided Pose Instructions

The system will guide you through different poses:

**Sample 1:**
```
Face the camera directly and stay still...
Capturing in 3 seconds...
  3...
  2...
  1...
Capturing frames... (hold position for 3 seconds)
```

**Sample 2:**
```
Turn your head SLIGHTLY to the LEFT and hold...
Capturing in 3 seconds...
  3...
  2...
  1...
Capturing frames... (hold position for 3 seconds)
```

**Sample 3 (if requested):**
```
Turn your head SLIGHTLY to the RIGHT and hold...
```

#### Step 5: Restart System

After updating, restart the system to activate the improved recognition:

```bash
docker-compose restart face-recognition
```

### Example: Before and After

**Before Update (1 embedding):**
```
SungHwan (0.52)  ← Low similarity score
```

**After Update (3 embeddings):**
```
SungHwan (0.82)  ← Much higher similarity score!
```

### When to Update Embeddings

Update embeddings when:
- Recognition similarity score is consistently below 0.65
- Person is often misidentified as "Unknown"
- Person's appearance has changed (glasses, facial hair, etc.)
- Lighting conditions are different from registration

### Tips for Best Update Results

1. **Vary Your Poses:**
   - Turn head slightly left (not extreme)
   - Turn head slightly right (not extreme)
   - Look directly at camera
   - Slight tilt up or down

2. **Maintain Good Lighting:**
   - Well-lit face (no shadows)
   - Avoid backlighting

3. **Stay Still During Capture:**
   - Hold each pose for the full 3 seconds
   - Avoid moving or talking

4. **Keep Same Person:**
   - Only the registered person should be in frame
   - System will verify it's the same person by comparing with existing embeddings

### Advanced: Direct Update Command

If you prefer to skip the helper script:

```bash
docker exec face_recognition_system python3 src/update_embeddings.py --name "PersonName" --samples 2
```

---

## Advanced: Direct Command

If you prefer, you can register directly using:

```bash
docker exec -it face_recognition_system python3 src/register_face.py --name "PersonName"
```

---

## Tips for Best Results

1. **Lighting**: Ensure the person's face is well-lit (avoid backlighting)
2. **Distance**: Stand 1-3 meters from the camera
3. **Angle**: Face the camera directly (avoid extreme angles)
4. **Stability**: Stay still for a few seconds during capture
5. **Expression**: Neutral expression works best

---

## Troubleshooting

### "Failed to detect any faces"
- Check if the person is in the camera view
- Improve lighting conditions
- Move closer to the camera
- Ensure the camera stream is working (check web viewer at http://localhost:8080)

### "All detected faces are already in the database"
- The person might already be registered under a different name
- Check the live video stream to see if they're being recognized
- Try having the person stand alone in the frame

### Registration shows wrong person
- If you accidentally registered the wrong face, you can:
  1. Delete the incorrect entry from the database
  2. Re-register with the improved script (it will filter out known faces)

---

## Database Location

Face database is stored at: `/app/data/face_database.pkl` (inside Docker container)

This is mapped to: `./data/face_database.pkl` (on your host machine)

---

## Current Registered Faces

To see who's currently registered, check the live video stream at:
http://localhost:8080

Registered faces will show their names in green boxes.
Unknown faces will show "Unknown" in red boxes.
