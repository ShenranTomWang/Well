source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

python -m data_gen.CancerMyth.prepare_dataset \
    generate \
        --out_dir ${SCRATCH_DIR}/datasets/CancerMyth

python -m data_gen.CancerMythNFP.prepare_dataset \
    generate \
        --out_dir ${SCRATCH_DIR}/datasets/CancerMythNFP

python -m data_gen.QA2.prepare_dataset \
    generate \
        --out_dir ${SCRATCH_DIR}/datasets

python -m data_gen.SynQA2.prepare_dataset \
    generate \
        --out_dir ${SCRATCH_DIR}/datasets

python -m data_gen.CREPE.prepare_dataset \
    --input_dir ${SCRATCH_DIR}/datasets/CREPE \
    --out_dir ${SCRATCH_DIR}/datasets