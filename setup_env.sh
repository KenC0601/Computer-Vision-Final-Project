ENV_DIR=".venv"

echo "Setting up Python virtual environment in $ENV_DIR..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$ENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate environment
echo "Activating environment..."
source "$ENV_DIR/bin/activate"

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

echo "----------------------------------------------------------------"
echo "✅ Environment setup complete."
echo ""
echo "To activate the environment, run:"
echo "source $ENV_DIR/bin/activate"
echo "----------------------------------------------------------------"
