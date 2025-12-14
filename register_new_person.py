#!/usr/bin/env python3
"""
Face Registration Helper
Simple, guided face registration process
"""

import sys
import subprocess
from pathlib import Path

def print_banner():
    print("\n" + "="*70)
    print("  FACE REGISTRATION HELPER")
    print("="*70)

def print_instructions():
    print("\n" + "-"*70)
    print("  INSTRUCTIONS:")
    print("-"*70)
    print("""
  1. Make sure the person you want to register is standing in front
     of the camera RIGHT NOW

  2. IMPORTANT: If other people are also in the camera view, that's OK!
     The system will:
     - Detect all faces in the camera
     - Identify which faces are already registered (KNOWN)
     - Show you only the UNKNOWN faces
     - Ask you to select which face to register

  3. For best results:
     - Face the camera directly
     - Ensure good lighting
     - Stay still for a few seconds

  4. The system will capture the best quality face automatically
""")
    print("-"*70)

def get_person_name():
    print("\n")
    while True:
        name = input("Enter the name of the person to register (or 'q' to quit): ").strip()

        if name.lower() == 'q':
            print("\nRegistration cancelled.")
            sys.exit(0)

        if not name:
            print("Name cannot be empty. Please try again.")
            continue

        if len(name) < 2:
            print("Name must be at least 2 characters. Please try again.")
            continue

        # Confirm the name
        print(f"\nYou entered: {name}")
        confirm = input("Is this correct? (y/n): ").strip().lower()

        if confirm == 'y':
            return name
        else:
            print("Let's try again...\n")

def run_registration(name):
    print("\n" + "="*70)
    print(f"  STARTING REGISTRATION FOR: {name}")
    print("="*70)
    print("\nPlease wait while the system:")
    print("  1. Connects to the camera")
    print("  2. Detects faces")
    print("  3. Checks against existing database")
    print("  4. Prompts you to confirm the face to register")
    print("\n")

    # Run the registration script
    try:
        # Use docker exec to run inside the container
        cmd = [
            "docker", "exec", "-it", "face_recognition_system",
            "python3", "src/register_face.py",
            "--name", name
        ]

        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n" + "="*70)
            print("  REGISTRATION SUCCESSFUL!")
            print("="*70)
            print(f"\n  {name} has been added to the face database.")
            print("  You can now see their name in the live video stream.")
            print("\n" + "="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("  REGISTRATION FAILED")
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

    name = get_person_name()
    run_registration(name)

if __name__ == "__main__":
    main()
