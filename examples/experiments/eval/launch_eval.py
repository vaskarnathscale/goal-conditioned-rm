# Adept from NeMo-Skills/pipeline/run_eval.py
# Mostly generate the eval script and then launch it through Scale train
import os
import subprocess
import sys
import tempfile
import uuid
from argparse import ArgumentParser
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).parents[0]))

SCRIPT_HELP = """
This script can be used to run evaluation of a model on a set of benchmarks.
It can run both greedy decoding and sampling and can parallelize generations
across multiple nodes. It uses nemo_skills/inference/generate_solutions.py
to generate solutions and nemo_skills/evaluation/evaluate_results.py to
evaluate them. It will set reasonable defaults for most of the generation parameters,
but you can override any of them by directly providing corresponding arguments
in the Hydra format.
"""


def get_server_command(server_type, num_gpus):
    num_tasks = num_gpus
    if server_type == "nemo":
        server_start_cmd = (
            f"(python /code/nemo_skills/inference/server/serve_nemo.py gpt_model_file=/model trainer.devices={num_gpus} "
            f"tensor_model_parallel_size={num_gpus} sandbox.sandbox_type=k8s > /opt/ml/checkpoints/server_logs.txt &)"
        )
        num_tasks = 1
    else:
        server_start_cmd = (
            f"(mpirun -np {num_gpus} --allow-run-as-root --oversubscribe python /code/nemo_skills/inference/server/serve_trt.py "
            "--model_path /model > /opt/ml/checkpoints/server_logs.txt &)"
        )
        num_tasks = 1  # we launch via mpirun directly

    return server_start_cmd, num_tasks


def get_greedy_cmd(benchmark, output_name="output-greedy.jsonl", extra_arguments=""):
    cmd = (
        f"echo 'Evaluating benchmark {benchmark}' && \\\n"
        "python nemo_skills/inference/generate_solutions.py \\\n"
        f"    server.server_type={{server_type}} \\\n"
        "    sandbox.sandbox_type=k8s \\\n"
        "    prompt.context_type=empty \\\n"
        f"    +dataset={benchmark} \\\n"
        f"    output_file=/opt/ml/checkpoints/{benchmark}/{output_name} \\\n"
        f"    {extra_arguments} && \\\n"
        "python nemo_skills/evaluation/evaluate_results.py \\\n"
        f"    prediction_jsonl_files=/opt/ml/checkpoints/{benchmark}/{output_name} \\\n"
        "    sandbox.sandbox_type=k8s && \\\n"
    )
    return cmd


def get_sampling_cmd(benchmark, random_seed, extra_arguments=""):
    extra_arguments = (
        f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    )
    return get_greedy_cmd(
        benchmark, output_name=f"output-rs{random_seed}.jsonl", extra_arguments=extra_arguments
    )


# # default number of samples for majority voting
BENCHMARKS = {
    "gsm8k": 8,
    "math": 4,
}

SLURM_CMD = """
nvidia-smi && \
cd /code && \
export PYTHONPATH=$PYTHONPATH:/code && \
{server_start_cmd} && \
if [ $SLURM_LOCALID -eq 0 ]; then \
    echo "Waiting for the server to start" && \
    tail -n0 -f /opt/ml/checkpoints/server_logs.txt | sed '/Running on all addresses/ q' && \
    {eval_cmds} true && \
    pkill python; \
    exit 0; \
else \
    sleep infinity; \
fi \
"""

# MOUNTS = "{NEMO_SKILLS_CODE}:/code,{model_path}:/model,{output_dir}:/results"
# JOB_NAME = "eval-{model_name}"

if __name__ == "__main__":
    parser = ArgumentParser(usage=SCRIPT_HELP + "\n\nscript arguments:\n\n" + "Find more ")
    wrapper_args = parser.add_argument_group("wrapper arguments")
    wrapper_args.add_argument("--model_path", required=True)
    wrapper_args.add_argument(
        "--server_type", choices=("nemo", "tensorrt_llm"), default="tensorrt_llm"
    )
    wrapper_args.add_argument("--num_gpus", type=int, required=True)
    wrapper_args.add_argument("--starting_seed", type=int, default=0)
    wrapper_args.add_argument(
        "--benchmarks",
        nargs="+",
        default=[],
        help="Need to be in a format <benchmark>:<num samples for majority voting>. "
        "Use <benchmark>:0 to only run greedy decoding.",
    )

    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    args.model_path = Path(args.model_path).absolute()

    server_start_cmd, num_tasks = get_server_command(args.server_type, args.num_gpus)

    format_dict = {
        "model_path": args.model_path,
        "model_name": args.model_path.name,
        "num_gpus": args.num_gpus,
        "server_start_cmd": server_start_cmd,
        "server_type": args.server_type,
    }

    # Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # if benchmarks are specified, only run those
    if args.benchmarks:
        BENCHMARKS = {k: int(v) for k, v in [b.split(":") for b in args.benchmarks]}

    eval_cmds = [
        get_greedy_cmd(benchmark, extra_arguments=extra_arguments)
        for benchmark in BENCHMARKS.keys()
    ]
    eval_cmds += [
        get_sampling_cmd(benchmark, rs, extra_arguments=extra_arguments)
        for benchmark, rs_num in BENCHMARKS.items()
        for rs in range(args.starting_seed, args.starting_seed + rs_num)
    ]

    # splitting eval cmds equally across num_nodes nodes
    eval_cmd = " ".join(eval_cmds)

    cmd = SLURM_CMD.format(**format_dict, eval_cmds=eval_cmd.format(**format_dict))

    cmd = cmd.strip()
    cmd = (
        f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(format_dict['num_gpus'])))} && "
        f"export SLURM_LOCALID=0 && "
        f"{cmd}"
    )
    with tempfile.NamedTemporaryFile(mode="wt", delete=False, dir=".") as fp:
        fp.write(cmd)
    print(f"Starting script: {fp.name}")
    print("Running command: \n", cmd)

    # Build image
    container_name = f"692474966980.dkr.ecr.us-west-2.amazonaws.com/content-understanding:nemo-eval-{str(uuid.uuid4())[:8]}"
    build_script_path = Path(__file__).parents[0] / "build_and_push_eval_image.sh"
    build_args = [
        "bash",
        str(build_script_path),
        container_name,  # image tag
        f"./{fp.name.split('/')[-1]}",  # starter script path
        f"./{str(args.model_path.relative_to(str(Path(__file__).parent)))}",  # model path
    ]
    # construct build args
    try:
        # Run the bash wrapper script
        subprocess.run(build_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during building: {e}")
        raise e
    finally:
        os.remove(fp.name)

    # launch through train
    train_cmd = ["scale-train", "train", "--image", container_name]
    try:
        # Run the bash wrapper script
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during building: {e}")
        raise e

# python launch_eval.py\
#   --model_path /home/ubuntu/workspace/cache/dense-rewards/models/ppo/nemo_models/gcp-llama-7b-gamma-1.0-alpha-1.0-lambd-095-ptx-002-1-epoch-correct-2-epochs-4096-goal-strategy-advantage_per_token_dense_reward_adjustment-window-size-15-rm-contrastive_codellama7b_v3_highest_perf_3_13_gse.nemo\
#   --server_type nemo \
#   --benchmarks gsm8k:0 \
#   --num_gpus 8 \
#   +prompt=code_sfted \
#   ++prompt.num_few_shots=0 \
#   ++split_name=test || true \
