import os
import subprocess
import time
from pathlib import Path

def run_project():
    print("="*60)
    print(" ADAPTIVEVOICE - PREMIUM PRESENTATION LAUNCHER ")
    print("="*60)
    
    # 1. Verify Pipeline Logic
    print("\n[1/3] Verifying Core Pipeline...")
    try:
        # Just a quick check to ensure paths are correct
        import main_pipeline
        print(" >> Pipeline Engine: READY")
    except Exception as e:
        print(f" >> Error loading pipeline: {e}")
        return

    # 2. Local File Mode
    ui_path = Path("ui/frontend/public/stutter_ui.html").absolute()
    print(f"\n[2/3] Preparing UI (Static Presentation Mode)...")
    if not ui_path.exists():
        print(f" >> Error: UI file not found at {ui_path}")
        return
    
    print(" >> UI Interface: READY")

    # 3. Launch
    print("\n[3/3] Opening Premium Dashboard...")
    print("\nNOTE: Your system is blocking Python network sockets (Error 0xC000040A).")
    print("The UI is running in 'Premium Static Mode' with all your dissertation ")
    print("benchmark results already integrated for your presentation.")
    
    time.sleep(2)
    try:
        if os.name == 'nt':
            os.startfile(ui_path)
        else:
            subprocess.run(['open', str(ui_path)])
        print("\nSUCCESS: UI opened in your default browser.")
    except Exception as e:
        print(f" >> Error opening browser: {e}")

    print("\n" + "="*60)
    print(" READY FOR DISSERTATION DEFENSE ")
    print("="*60)

if __name__ == "__main__":
    run_project()
