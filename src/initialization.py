import os

def setup_project_structure():
    """
    Setup project structure
    """
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write("__pycache__/\n.ipynb_checkpoints/\n*.pyc\n.DS_Store\n")
    print("Initialization complete: Checked project structure.")
