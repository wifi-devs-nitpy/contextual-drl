rsync -rv  \
    --exclude=".venv/" \
    --exclude="*.png" \
    --exclude="*.npy" \
    --exclude="*.json" \
    --exclude="__pycache__/" \
    "/mnt/c/Users/jomon/Documents/wifi_9/mapc-cmab/" \
    "$HOME/documents/wifi9/mapc-cmab/"


