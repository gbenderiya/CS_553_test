#!/bin/bash

# Variables
PORT=22008
MACHINE="paffenroth-23.dyn.wpi.edu"
STUDENT_ADMIN_KEY_PATH="./keys" # Update this when checking the keys in the correct location
PROJECT_DIR="CS553_assignment1"
REMOTE_PROJECT_PATH="~/project/$PROJECT_DIR"
SSH_PATH="~/.ssh"
NEW_KEY_PATH="$SSH_PATH/my_key"
NEW_KEY_PUB_PATH="$SSH_PATH/my_key.pub"

# Step 1: Generate new SSH keys
echo "Generating a new SSH key..."
ssh-keygen -f $NEW_KEY_PATH -t ed25519 -N ""

# Step 2: Copy the new key to authorized_keys on the remote VM
echo "Updating authorized_keys on the VM..."
ssh -i $STUDENT_ADMIN_KEY_PATH/student-admin_key -p $PORT -o StrictHostKeyChecking=no student-admin@$MACHINE "echo 'Updating keys...' && mkdir -p $SSH_PATH && cat > $SSH_PATH/authorized_keys" < $NEW_KEY_PUB_PATH

# Step 3: Add the new key to the SSH agent
eval "$(ssh-agent -s)"
ssh-add $NEW_KEY_PATH
the
# Step 4: Verify  new key works
ssh -i $NEW_KEY_PATH -p $PORT -o StrictHostKeyChecking=no student-admin@$MACHINE "echo 'New key works!'"

# Step 5: Redeploy the application
echo "Redeploying the application..."
ssh -i $NEW_KEY_PATH -p $PORT -o StrictHostKeyChecking=no student-admin@$MACHINE <<EOF
    cd $REMOTE_PROJECT_PATH
    nohup venv/bin/python3 app.py > log.txt 2>&1 &
    echo "App started!"
EOF

echo "Recovery process complete!"
