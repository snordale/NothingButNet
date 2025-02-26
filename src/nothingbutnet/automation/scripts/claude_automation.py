import os
import sys
import logging
import anthropic
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import *

class ClaudeAutomation:
    def __init__(self):
        self.setup_logging()
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
    def setup_logging(self):
        """Configure logging for the automation system"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ClaudeAutomation")
        
    def load_prompt_template(self, task_type):
        """Load prompt template for the given task type"""
        prompt_file = PROMPTS_DIR / f"{task_type}.txt"
        try:
            with open(prompt_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Prompt template not found for task type: {task_type}")
            return None
            
    def get_project_context(self):
        """Gather current project context"""
        context = {}
        
        # Read main data collector file
        try:
            with open(PROJECT_PATHS["data_collector"], 'r') as f:
                context["data_collector"] = f.read()
        except Exception as e:
            self.logger.error(f"Error reading data collector: {e}")
            
        # Add other context as needed
        return context
        
    def execute_task(self, task_type):
        """Execute a specific automation task"""
        self.logger.info(f"Executing task: {task_type}")
        
        # Load prompt template
        prompt = self.load_prompt_template(task_type)
        if not prompt:
            return
            
        # Get project context
        context = self.get_project_context()
        
        # Format prompt with context
        formatted_prompt = prompt.format(**context)
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": formatted_prompt
                }]
            )
            
            # Process and save response
            self.save_response(task_type, response)
            
        except Exception as e:
            self.logger.error(f"Error executing task {task_type}: {e}")
            
    def save_response(self, task_type, response):
        """Save Claude's response to output directory"""
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUTS_DIR / f"{task_type}_{timestamp}.txt"
        
        try:
            with open(output_file, 'w') as f:
                f.write(response.content[0].text)
            self.logger.info(f"Saved response to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving response: {e}")
            
def main():
    if len(sys.argv) != 2:
        print("Usage: python claude_automation.py <task_type>")
        sys.exit(1)
        
    task_type = sys.argv[1]
    if task_type not in TASK_TYPES:
        print(f"Invalid task type. Available tasks: {list(TASK_TYPES.keys())}")
        sys.exit(1)
        
    automation = ClaudeAutomation()
    automation.execute_task(task_type)

if __name__ == "__main__":
    main() 