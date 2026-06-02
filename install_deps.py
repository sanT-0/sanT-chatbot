import subprocess
import sys
import os

def install_requirements():
    """Installs packages from requirements.txt using pip."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(req_path):
        print('[WARN] requirements.txt not found.')
        sys.exit(1)
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_path])
        print('[OK] All dependencies installed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] Failed to install dependencies: {e}')
        sys.exit(e.returncode)

if __name__ == '__main__':
    install_requirements()
