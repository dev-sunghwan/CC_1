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
