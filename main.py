import argparse
import os
import re
import shutil
import subprocess
import sys
from importlib.util import find_spec
from threading import Thread

from btgen_api_version import version
from btgenapi.repositories_versions import btgen_commit_hash


sys.path.append(os.path.dirname(os.path.realpath(__file__)))



os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

python = sys.executable
default_command_live = True
index_url = os.environ.get('INDEX_URL', "")
re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")

btgen_name = 'btgen'

btgen_github_repo = 'https://github.com/oldhand7/stable_diffusion_img_gen'

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = modules_path
dir_repos = "repositories"


def onerror(func, path, exc_info):
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise 'Failed to invoke "shutil.rmtree", git management failed.'



def git_clone(url, dir, name, hash=None):
    import pygit2

    try:
        try:
            repo = pygit2.Repository(dir)
            remote_url = repo.remotes['origin'].url
            if remote_url not in [btgen_github_repo]:
                del repo
                raise url
            else:
                print(f'{name} exists and URL is correct.')
            url = remote_url
        except:
            if os.path.isdir(dir) or os.path.exists(dir):
                sys.exit(1)
            os.makedirs(dir, exist_ok=True)
            repo = pygit2.clone_repository(url, dir)
            print(f'{name} cloned from {url}.')

        remote = repo.remotes['origin']
        remote.fetch()

        commit = repo.get(hash)

        repo.checkout_tree(commit, strategy=pygit2.GIT_CHECKOUT_FORCE)
        repo.set_head(commit.id)

    except Exception as e:
        raise e


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""



def run_pip(command, desc=None, live=default_command_live):
    try:
        index_url_line = f' --index-url {index_url}' if index_url != '' else ''
        return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}",
                   errdesc=f"Couldn't install {desc}", live=live)
    except Exception as e:
        return None



def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all requirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    import importlib.metadata
    import packaging.version

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "":
                continue

            m = re.match(re_requirement, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            if version_required == "":
                continue

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                return False

    return True


def download_repositories():
    import pygit2
    import requests

    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')

    if http_proxy is not None:
        os.environ['http_proxy'] = http_proxy

    if https_proxy is not None:
        os.environ['https_proxy'] = https_proxy

    try:
        requests.get("https://policies.google.com/privacy", timeout=5)
        btgen_repo_url = btgen_github_repo
    except:
        btgen_repo_url = btgen_github_repo
    btgen_repo = os.environ.get(
        'btgen_repo', btgen_repo_url)
    git_clone(btgen_repo, repo_dir(btgen_name),
              "btgen", btgen_commit_hash)


def is_installed(package):
    try:
        spec = find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None




def install_dependents(args):
    if not args.skip_pip:
        torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")

        # Check if you need pip install
        requirements_file = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
        if not requirements_met(requirements_file):
            run_pip(f"install -r \"{requirements_file}\"", "requirements")

        if not is_installed("torch") or not is_installed("torchvision"):
            run_pip(f"install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}", "torch")

        if args.persistent and not is_installed("sqlalchemy"):
            run_pip(f"install sqlalchemy==2.0.25", "sqlalchemy")

    skip_sync_repo = False
    if args.sync_repo is not None:
        if args.sync_repo == 'only':
            download_repositories()
            models_path = os.path.join(
                script_path, dir_repos, btgen_name, "models")
            return False
        elif args.sync_repo == 'skip':
            skip_sync_repo = True
        else:
            exit(1)

    # if not skip_sync_repo:
    #     download_repositories()

    # Add dependent repositories to import path
    sys.path.append(script_path)
    btgen_path = os.path.join(script_path, dir_repos, btgen_name)
    sys.path.append(btgen_path)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def prepare_environments(args) -> bool:
    if args.gpu_device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)

    if args.base_url is None or len(args.base_url.strip()) == 0:
        host = args.host
        if host == '0.0.0.0':
            host = '127.0.0.1'
        args.base_url = f"http://{host}:{args.port}"

    sys.argv = [sys.argv[0]]

    if args.preset is not None:
        # Remove and copy preset folder
        origin_preset_folder = os.path.abspath(os.path.join(script_path, dir_repos, btgen_name, 'presets'))
        preset_folder = os.path.abspath(os.path.join(script_path, 'presets'))
        if os.path.exists(preset_folder):
            shutil.rmtree(preset_folder)
        shutil.copytree(origin_preset_folder, preset_folder)

    import modules.config as config
    import btgenapi.parameters as parameters
    parameters.default_inpaint_engine_version = config.default_inpaint_engine_version
    parameters.default_styles = config.default_styles
    parameters.default_base_model_name = config.default_base_model_name
    parameters.default_refiner_model_name = config.default_refiner_model_name
    parameters.default_refiner_switch = config.default_refiner_switch
    parameters.default_loras = config.default_loras
    parameters.default_cfg_scale = config.default_cfg_scale
    parameters.default_prompt1 = config.default_prompt1
    parameters.default_aspect_ratio = "800*1120"
    parameters.available_aspect_ratios = [parameters.get_aspect_ratio_value(a) for a in config.available_aspect_ratios]

    # download_models()

    if args.preload_pipeline:
        preplaod_pipeline()

    # Init task queue
    import btgenapi.worker as worker
    from btgenapi.task_queue import TaskQueue
    worker.worker_queue = TaskQueue(queue_size=args.queue_size, hisotry_size=args.queue_history, webhook_url=args.webhook_url, persistent=args.persistent)

    return True


def pre_setup(skip_sync_repo: bool = False,
              disable_image_log: bool = False,
              skip_pip=False,
              load_all_models: bool = False,
              preload_pipeline: bool = False,
              always_gpu: bool = False,
              all_in_fp16: bool = False,
              preset: str | None = None):
    class Args(object):
        host = '127.0.0.1'
        port = 9999
        base_url = None
        sync_repo = None
        disable_image_log = False
        skip_pip = False
        preload_pipeline = False
        queue_size = 100
        queue_history = 0
        preset = None
        webhook_url = None
        persistent = False
        always_gpu = False
        all_in_fp16 = False
        gpu_device_id = None
        apikey = None


    args = Args()
    if skip_sync_repo:
        args.sync_repo = 'skip'
    args.disable_image_log = disable_image_log
    args.skip_pip = skip_pip
    args.preload_pipeline = preload_pipeline
    args.always_gpu = always_gpu
    args.all_in_fp16 = all_in_fp16
    args.preset = preset

    sys.argv = [sys.argv[0]]
    if args.preset is not None:
        sys.argv.append('--preset')
        sys.argv.append(args.preset)

    if args.disable_image_log:
        sys.argv.append('--disable-image-log')

    install_dependents(args)

    import btgenapi.args as _
    prepare_environments(args)

    if load_all_models:
        import modules.config as config
        from btgenapi.parameters import default_inpaint_engine_version
        config.downloading_upscale_model()
        config.downloading_inpaint_models(default_inpaint_engine_version)
        config.downloading_controlnet_canny()
        config.downloading_controlnet_cpds()
        config.downloading_ip_adapters()


def preplaod_pipeline():
    import modules.default_pipeline as _


if __name__ == "__main__":

    
    from btgenapi.base_args import add_base_args
    parser = argparse.ArgumentParser()
    add_base_args(parser, True)

    args, _ = parser.parse_known_args()
    install_dependents(args)

    from btgenapi.args import args
    from repositories.btgen.launch import download_models
    download_models()
    if prepare_environments(args):
        sys.argv = [sys.argv[0]]

        # Load pipeline in new thread
        preload_pipeline_thread = Thread(target=preplaod_pipeline, daemon=True)
        preload_pipeline_thread.start()

        # Start task schedule thread
        from btgenapi.worker import task_schedule_loop
        task_schedule_thread = Thread(target=task_schedule_loop, daemon=True)
        task_schedule_thread.start()

        # Start api server
        from btgenapi.api import start_app

        start_app(args)
