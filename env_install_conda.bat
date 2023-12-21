conda env create --file environment.yml
call activate inpainting

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pause