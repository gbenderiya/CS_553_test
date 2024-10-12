#! /bin/bash

PORT=22008
MACHINE=paffenroth-23.dyn.wpi.edu
PROJECT_DIR="CS553_assignment1"
TMP_DIR="tmp"

# Change to the temporary directory
cd $TMP_DIR

# Add the key to the ssh-agent
eval "$(ssh-agent -s)"
ssh-add my_key

echo "SSH Agent status:"
ssh-add -l

# commands
ssh -i tmp/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "cd project"
ssh -i tmp/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "ls ${PROJECT_DIR}"
ssh -i tmp/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "sudo apt install -qq -y python3-venv && echo 'venv installed successfully' || echo 'venv installation failed'"
ssh -i tmp/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "cd ${PROJECT_DIR} && python3 -m venv venv"
ssh -i tmp/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "cd ${PROJECT_DIR} && source venv/bin/activate && pip install -r requirements.txt"
ssh -i tmp/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "nohup ${PROJECT_DIR}/venv/bin/python3 ${PROJECT_DIR}/app.py > log.txt 2>&1 &"
ssh -i tmp/my_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "tail -n 20 project/${PROJECT_DIR}/log.txt"


# nohup ./whatever > /dev/null 2>&1 

# debugging ideas
# sudo apt-get install gh
# gh auth login
# requests.exceptions.HTTPError: 429 Client Error: Too Many Requests for url: https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/v1/chat/completions
# log.txt
