#!/bin/bash

# Variables
PORT=22008
MACHINE="paffenroth-23.dyn.wpi.edu"
STUDENT_ADMIN_KEY_PATH="/mnt/c/Users/ganer/Documents/github/mlops/keys"
SSH_PATH="/mnt/c/Users/ganer/"
REPO_URL="https://github.com/gbenderiya/CS553_assignment1" 
PROJECT_DIR="CS553_assignment1"
TMP_DIR="tmp"
REMOTE_PROJECT_PATH="~/project"

# Step 0: Check if connection works with student-admin_key
ssh -i student-admin_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "echo 'SSH connection works student-admin_key'"

# Step 1: Clean up known_hosts and previous runs
echo "Cleaning up previous runs and known_hosts..."
ssh-keygen -f "${SSH_PATH}.ssh/known_hosts" -R "[${MACHINE}]:${PORT}"
rm -rf $TMP_DIR

# Step 2: Set up temporary directory and copy keys
echo "Setting up temporary directory and copying keys..."
mkdir $TMP_DIR
echo "Listing keys directory:"
ls ${STUDENT_ADMIN_KEY_PATH}

echo "Attempting to copy keys:"
cp "${STUDENT_ADMIN_KEY_PATH}/student-admin_key" $TMP_DIR
cp "${STUDENT_ADMIN_KEY_PATH}/student-admin_key.pub" $TMP_DIR

# Step 3: Set permissions for the key
cd $TMP_DIR
chmod 600 student-admin_key*

# Step 4: Generate a new key
echo "Generating a new SSH key..."
rm -f my_key*
ssh-keygen -f my_key -t ed25519 -N "team8gry"

# Step 5: Update authorized_keys locally
cat my_key.pub > "${SSH_PATH}.ssh/authorized_keys"
cat student-admin_key.pub >> "${SSH_PATH}.ssh/authorized_keys"
chmod 600 "${SSH_PATH}.ssh/authorized_keys"


# Step 6: Display authorized_keys for verification
echo "Verifying local authorized_keys file:"
ls -l "${SSH_PATH}.ssh/authorized_keys"
cat "${SSH_PATH}.ssh/authorized_keys"

# Step 7: Copy the authorized_keys to the server
echo "Copying authorized_keys to the remote server..."
scp -i student-admin_key -P ${PORT} -o StrictHostKeyChecking=no authorized_keys student-admin@${MACHINE}:~/.ssh/

# Step 8: Add the key to ssh-agent
echo "Adding key to ssh-agent..."
eval "$(ssh-agent -s)"
ssh-add my_key

echo "SSH Agent status:"
ssh-add -l

# Step 9: Verify the key file on the server
echo "Verifying the authorized_keys on the server..."
ssh -i student-admin_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "cat ~/.ssh/authorized_keys"

# Step 10: Check if the project folder exists on the server, create it if it doesn't
ssh -i student-admin_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "echo 'SSH connection works'"
echo "Checking if the project directory exists on the server..."
ssh -i student-admin_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "mkdir -p ${REMOTE_PROJECT_PATH}"

# Step 11: Check if connection works with my_key
ssh -i ${TMP_DIR}/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "echo 'SSH connection works with my_key'"

# Step 12: Clone the repository locally
echo "Cloning the repository to local machine..."
git clone ${REPO_URL}

# Step 13: Copy the repository to the project folder on the server
echo "Copying the project files to the server project directory..."
scp -i student-admin_key -P ${PORT} -o StrictHostKeyChecking=no -r ${PROJECT_DIR} student-admin@${MACHINE}:${REMOTE_PROJECT_PATH}/
ssh -i student-admin_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "ls -al ${REMOTE_PROJECT_PATH}/${PROJECT_DIR} || echo 'Directory not found'"

# Final message
echo "Deployment completed successfully!"
