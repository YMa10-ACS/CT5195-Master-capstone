if [ ! -d ".venv" ]; then
	python3 -m venv .venv
fi

source .venv/bin/activate
pip install numpy
pip install torch
pip install psutil
pip install flask
pip install transformers
pip install torchvision
pip install einops
pip install requests

if [ ! -d "../nanoVLM" ]; then
	git clone https://github.com/huggingface/nanoVLM.git
	mv nanoVLM ../
fi

