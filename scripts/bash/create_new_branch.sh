#!/bin/bash
# chmod +x ./scripts/bash/create_new_branch.sh
# ./scripts/bash/create_new_branch.sh feature/new_branch branch_based_on
#   - example: ./scripts/bash/create_new_branch.sh feature/implementing_sagemaker_models main

# Extract new branch name
NEW_BRANCH_NAME=$1

# Extract branch to base new branch on
BASE_BRANCH_NAME=$2

# Check if new branch name is provided
if [ -z "$NEW_BRANCH_NAME" ]; then
    echo "Please provide a new branch name"
    exit 1
fi

# Check if base branch name is provided
if [ -z "$BASE_BRANCH_NAME" ]; then
    echo "Please provide a base branch name"
    exit 1
fi

# Pull latest changes from base branch
git checkout ${BASE_BRANCH_NAME}
git pull origin ${BASE_BRANCH_NAME}

# Create new branch
git checkout -b ${NEW_BRANCH_NAME}

# Push to remote repository
git push -u origin ${NEW_BRANCH_NAME}
