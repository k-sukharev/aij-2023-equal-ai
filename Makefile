extract: extract_videos extract_annotations

extract_videos:
	7z x input/slovo.zip -oinput slovo/*.*


	7z x input/slovo_annotations.zip -oinput

extract_baseline:
	7z x input/baseline.zip -oinput/baseline

run:
	mkdir -p working/
	cd working/; \
	export HYDRA_FULL_ERROR=1; \
	poetry run python ../scripts/run.py hydra.job.chdir=False

run_docker:
	docker run --gpus all --shm-size=16g --rm -it -v $$(pwd):/workspace -w /workspace/working aij-skhrv:latest \
	bash -c "export HYDRA_FULL_ERROR=1; export PYTHONPATH="$PYTHONPATH:/workspace/src"; python ../scripts/run.py hydra.job.chdir=False"

notebook:
	mkdir -p notebooks; cd notebooks; poetry run jupyter notebook

notebook_docker:
	mkdir -p notebooks; docker run --gpus all --shm-size=16g --rm -it -p 8888:8888 -v $$(pwd):/workspace -w /workspace/notebooks aij-skhrv:latest jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=8888

tensorboard:
	poetry run tensorboard --logdir working/

build:
	docker build -t aij-skhrv:latest .

pretrain:
	docker run --gpus all --shm-size=16g --rm -it -v $$(pwd):/workspace -w /workspace/VideoMAEv2 aij-skhrv:latest bash scripts/pretrain/vit_n_slovo_pt.sh

finetune:
	docker run --gpus all --shm-size=32g --rm -it -v $$(pwd):/workspace -w /workspace/VideoMAEv2 aij-skhrv:latest bash scripts/finetune/x3d_m_slovo_ft.sh

distill:
	docker run --gpus all --shm-size=32g --rm -it -v $$(pwd):/workspace -w /workspace/input/mmaction aij-skhrv:latest mim train mmrazor configs/kd_logits_mvit_x3d_slovo.py --launcher pytorch --gpus 2

.PHONY: extract extract_videos extract_annotations run run_docker notebook notebook_docker tensorboard build pretrain finetune distill
