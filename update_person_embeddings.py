#!/usr/bin/env python3
"""
Update Person Embeddings Helper
Simple, guided process to add more face embeddings to existing people
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    print("\n" + "="*70)
    print("  UPDATE PERSON EMBEDDINGS - IMPROVE RECOGNITION ACCURACY")
    print("="*70)


def print_instructions():
    print("\n" + "-"*70)
    print("  WHY UPDATE EMBEDDINGS?")
    print("-"*70)
    print("""
  By adding 2-3 more face samples for a person, the recognition system
  will become more accurate and robust to:
  - Different head angles (left, right, up, down)
  - Varying lighting conditions
  - Different facial expressions
  - Distance variations

  This will improve recognition similarity scores from ~0.50-0.60 to ~0.65-0.85!
""")
    print("-"*70)


def list_people_in_database():
    """List all people currently in the database"""
    try:
        # Run command to list database contents
        cmd = [
            "docker", "exec", "face_recognition_system",
            "python3", "-c",
            "import pickle; db = pickle.load(open('/app/data/face_database.pkl', 'rb')); "
            "print('\\nPeople in database:'); "
            "[print(f'  - {name} ({len(data.get(\"embeddings\", [data.get(\"embedding\")])) if \"embeddings\" in data or \"embedding\" in data else 0} embedding(s))') for name, data in db.items()]"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
        else:
            print("\nCould not load database. Make sure the system is running.")
            print("Run: docker-compose up -d")

    except FileNotFoundError:
        print("\nERROR: Docker is not running or container is not started.")
        print("Please run: docker-compose up -d")
    except Exception as e:
        print(f"\nERROR: {e}")


def get_person_name():
    print("\n")
    while True:
        name = input("Enter the name of the person to update (or 'q' to quit): ").strip()

        if name.lower() == 'q':
            print("\nUpdate cancelled.")
            sys.exit(0)

        if not name:
            print("Name cannot be empty. Please try again.")
            continue

        # Confirm the name
        print(f"\nYou entered: {name}")
        confirm = input("Is this correct? (y/n): ").strip().lower()

        if confirm == 'y':
            return name
        else:
            print("Let's try again...\n")


def get_num_samples():
    """Ask user how many additional samples to capture"""
    print("\n" + "-"*70)
    print("  HOW MANY ADDITIONAL SAMPLES?")
    print("-"*70)
    print("""
  Recommended: 2 samples (for a total of 3 embeddings)

  More samples = better accuracy, but diminishing returns after 3-4 samples.
""")
    print("-"*70)

    while True:
        try:
            samples = input("\nNumber of additional samples to capture (default: 2): ").strip()

            if not samples:
                return 2  # Default

            num = int(samples)

            if num < 1:
                print("Must capture at least 1 sample.")
                continue

            if num > 5:
                print("Warning: More than 5 samples may not significantly improve accuracy.")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue

            return num

        except ValueError:
            print("Please enter a valid number.")


def run_update(name, num_samples):
    print("\n" + "="*70)
    print(f"  UPDATING EMBEDDINGS FOR: {name}")
    print("="*70)
    print(f"\nWill capture {num_samples} additional sample(s)")
    print("\nPrepare to follow the on-screen instructions:")
    print("  1. Face the camera directly")
    print("  2. Turn slightly left when prompted")
    print("  3. Turn slightly right when prompted")
    print("\nPress Enter when ready...")
    input()

    # Run the update script
    try:
        cmd = [
            "docker", "exec", "-it", "face_recognition_system",
            "python3", "src/update_embeddings.py",
            "--name", name,
            "--samples", str(num_samples)
        ]

        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n" + "="*70)
            print("  UPDATE SUCCESSFUL!")
            print("="*70)
            print(f"\n  {name}'s face recognition has been improved!")
            print("\n  IMPORTANT: Restart the system to use the updated database:")
            print("    docker-compose restart face-recognition")
            print("\n" + "="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("  UPDATE FAILED")
            print("="*70)
            print("\n  Please check the error messages above and try again.")
            print("\n" + "="*70 + "\n")

    except FileNotFoundError:
        print("\nERROR: Docker is not running or container is not started.")
        print("Please make sure the face recognition system is running.")
        print("Run: docker-compose up -d")
    except Exception as e:
        print(f"\nERROR: {e}")


def main():
    print_banner()
    print_instructions()
    list_people_in_database()

    name = get_person_name()
    num_samples = get_num_samples()
    run_update(name, num_samples)


if __name__ == "__main__":
    main()
