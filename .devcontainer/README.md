# Dev Container Configuration

This workspace includes a dev container configuration that provides:

## Features
- **Python 3.13.5** development environment
- **Git** version control
- **GitHub CLI** for GitHub operations
- Pre-configured Python extensions for VS Code

## Included VS Code Extensions
- Python
- Pylance (Python language server)
- Flake8 (linting)
- Black Formatter
- Jupyter notebooks support

## Getting Started
1. Make sure you have Docker installed and running
2. Install the "Dev Containers" extension in VS Code
3. Open this workspace in VS Code
4. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select "Dev Containers: Reopen in Container"
5. VS Code will build and start the dev container

## Dependencies
- Add your Python dependencies to `requirements.txt`
- They will be automatically installed when the container is created

## Customization
- Edit `.devcontainer/devcontainer.json` to customize the environment
- Add additional VS Code extensions, settings, or features as needed
