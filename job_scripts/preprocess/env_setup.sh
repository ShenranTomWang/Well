uv venv $SCRATCH_DIR/envs/FP_Hallucination/.venv
source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
uv sync --active

python -c "import nltk; nltk.download('punkt_tab')"