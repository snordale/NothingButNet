#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from crontab import CronTab

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import SCHEDULE, TASK_TYPES

def setup_cron_jobs():
    """Set up cron jobs for each automation task"""
    # Get the absolute path to the automation script
    automation_script = Path(__file__).parent / "claude_automation.py"
    python_path = sys.executable
    
    try:
        # Initialize crontab for current user
        cron = CronTab(user=True)
        
        # Remove existing automation jobs
        cron.remove_all(comment="nba_automation")
        
        # Add new jobs for each task
        for task_type, schedule in SCHEDULE.items():
            job = cron.new(
                command=f"{python_path} {automation_script} {task_type}",
                comment=f"nba_automation_{task_type}"
            )
            job.setall(schedule)
            
        # Write the changes
        cron.write()
        print("Successfully set up cron jobs:")
        for job in cron:
            if "nba_automation" in job.comment:
                print(f"- {job.comment}: {job}")
                
    except Exception as e:
        print(f"Error setting up cron jobs: {e}")
        sys.exit(1)

def main():
    # Ensure ANTHROPIC_API_KEY is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        sys.exit(1)
    
    setup_cron_jobs()

if __name__ == "__main__":
    main() 