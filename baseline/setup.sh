#!/usr/bin/env bash
conda create -n retro python=3.5 -y
source activate retro
pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip install -r requirements.txt
pip install --no-deps git+https://github.com/fgvbrt/baselines.git@fbbd70a62e606ac8c9c92162f5bcf843dfd983ca
git clone https://github.com/openai/retro-contest.git && cd retro-contest/support && pip install .

# download roms
wget -qO - https://www.dropbox.com/s/8i0mh0bn2bbe1w5/roms.tar.gz?dl=0 | tar xzv
find ./roms/ -name 'Sonic*' -type d -exec python -m retro.import {} \;