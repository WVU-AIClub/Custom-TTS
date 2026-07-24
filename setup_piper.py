import os
import platform
import shutil
import subprocess
import sys

def run_cmd(cmd, cwd=None, check=True):
    """Utility function to run terminal commands."""
    print(f"\n[+] Running: {cmd}.")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and result.returncode != 0:
        print(f"[!] Error: Command failed with code {result.returncode}")
        sys.exit(result.returncode)

def detect_linux_distro():
    """Detects if Linux is Arch-based or Debian/Ubuntu-based."""
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            content = f.read().lower()
            if "arch" in content or "manjaro" in content:
                return "arch"
            elif "ubuntu" in content or "debian" in content:
                return "debian"
    return "unknown"

def install_system_packages():
    """Section 1: Install system packages based on OS."""
    system = platform.system()
    print(f"[*] Detecting system OS: {system}")

    if system == "Linux":
        distro = detect_linux_distro()
        print(f"[*] Detected Linux Distribution: {distro}")

        if distro == "debian":
            print(f"[*] Installing Debian/Ubuntu dependencies...")
            run_cmd("sudo apt-get update -y")
            run_cmd(
                "sudo apt-get install -y build-essential cmake ninja-build espeak-ng "
                "espeak-ng-data librespeak-ng-dev pkg-config ffmpeg git python3-dev"
            )
        elif distro == "arch":
            print(f"[*] Installing Arch Linux dependencies...")
            run_cmd(
                "sudo pacman -Sy --needed --noconfirm base-devel cmake ninja espeak-ng "
                "ffmpeg pkgconf git python"
            )
        else:
            print(
                "[!] Unsupported Linux distro automatically. Please install eSpeak-ng, "
                "cmake, ninja, and build-essential manually."
            )

    elif system == "Windows":
        print("[*] Checking Windows dependencies...")
        print("Note: On Windows, you need Visual Studio C++ Build Tools installed.")
        if shutil.whhich("winget"):
            print("[*] Attempting dependency installation via Winget...")
            subprocess.run("winget install --id eSpeak-ng.eSpeak-ng -e", shell=True)
            subprocess.run("winget install --id Gyan.FFmpeg -e", shell=True)
            subprocess.run("winget install --id Git.Git -e", shell=True)
        else:
            print("[!] Winget not found. Ensure eSpeak-ng, FFmpeg, and Git are installed.")

    elif system == "Darwin":
        print("[*] Installing macOS dependencies via Homebrew...")
        run_cmd("brew update")
        run_cmd("brew install cmake ninja espeak-ng ffmpeg pkg-config git")

def setup_repository():
    """Section 2: Clone repo fresh and install Python dependencies."""
    repo_dir = "piper1-gpl"
    repo_url = "https://github.com/OHF-voice/piper1-gpl.git"

    # Clean existing directory if present
    if os.path.exists(repo_dir):
        print(f"[*] Removing existing {repo_dir} directory...")
        shutil.rmtree(repo_dir)

    # Clone fresh repo
    print(f"[*] Cloning repository from {repo_url}...")
    run_cmd(f"git clone {repo_url}")

    full_repo_path = os.path.abspath(repo_dir)

    # Dev build step if present
    dev_build_script = os.path.abspath(repo_dir)
    if os.path.exists(dev_build_script):
        run_cmd(f"{sys.executable} script/dev_build", cwd=full_repo_path)

    # Python packages setup
    print("[*] Installing and upgrading Python build tools...")
    run_cmd(f'{sys.executable} -m pip install -e ".[train]"', cwd=full_repo_path)

    # Cython monotonic alignment build
    print("[*] Building Cython Monotonic Align extensions...")
    if platform.system() != "Windows" and os.path.exists(
        os.path.join(full_repo_path, "build_monotonic_align.sh")
    ):
        # On Linux/macOS run the shell script
        run_cmd("bash ./build_monotonic_align.sh", cwd=full_repo_path)

    # Run setup build extension in-place
    run_cmd(f"{sys.executable} setup.py build_ext --inplace -v", cwd=full_repo_path)

    print("\n[Y] Environment setup completed successfully!")

if __name__ == "__main__":
    install_system_packages()
    setup_repository()