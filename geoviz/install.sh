#!/bin/bash
# Installation script for Geometry Visualizer

set -e

echo "================================"
echo "Geometry Visualizer Setup"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.11+ is required (found Python $python_version)"
    exit 1
fi
echo "✓ Python $python_version found"
echo ""

# Check Node.js version
echo "Checking Node.js version..."
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi
node_version=$(node --version | cut -d'v' -f2 | cut -d. -f1)
if [ "$node_version" -lt 18 ]; then
    echo "Error: Node.js 18+ is required (found Node.js $node_version)"
    exit 1
fi
echo "✓ Node.js $(node --version) found"
echo ""

# Install backend
echo "Installing Python backend..."
cd backend
pip install -e ".[dev]"
cd ..
echo "✓ Backend installed"
echo ""

# Install frontend
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..
echo "✓ Frontend dependencies installed"
echo ""

echo "================================"
echo "Installation complete!"
echo "================================"
echo ""
echo "To start the backend tests:"
echo "  cd backend && pytest tests/"
echo ""
echo "To start the frontend viewer:"
echo "  cd frontend && npm run dev"
echo ""
echo "To use the Python library:"
echo "  python3"
echo "  >>> from geometry_visualizer.primitives import Point2D"
echo "  >>> from geometry_visualizer.algorithms.convex_hull import convex_hull_2d"
echo ""
