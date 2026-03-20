#!/bin/bash
set -e

# Deploy to HuggingFace Spaces
# Adds YAML frontmatter to README.md on a temporary branch, pushes, then cleans up.

# Ensure clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: Working tree not clean. Commit or stash your changes first."
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Create temporary deploy branch
git checkout -b _hf_deploy

# Replace README.md with HF Spaces metadata
cat > README.md << 'EOF'
---
title: ReView
emoji: 📚
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: interface/Demo.py
pinned: true
license: mit
short_description: Visualize and analyze scientific peer reviews
---
EOF

# Commit and push
git add README.md
git commit -m "Add HF Spaces metadata for deployment"
git push space _hf_deploy:main --force

# Clean up
git checkout "$CURRENT_BRANCH"
git branch -D _hf_deploy

echo "Deployed to HF Spaces successfully!"
