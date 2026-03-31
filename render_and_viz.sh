#!/usr/bin/env bash
# render_and_viz.sh — Automate results processing and documentation rendering.

set -e

echo "─────────────────────────────────────────────────"
echo "🎨 Results Processing & Visualization"
echo "─────────────────────────────────────────────────"

# 1. Generate Verification Viz
echo "🔍 1. Generating label verification images..."
mkdir -p verification_viz
python3 src/visualize_labels.py \
    --images ./dataset_ran_ado/train/images \
    --labels ./dataset_ran_ado/train/labels \
    --out ./verification_viz \
    --num 20

# 2. Render Quarto Website
echo "📄 2. Rendering Quarto website..."
quarto render

echo "─────────────────────────────────────────────────"
echo "✅ Done! View results in:"
echo "   - Local Site: _site/index.html"
echo "   - Label Viz:  verification_viz/"
echo "─────────────────────────────────────────────────"
