


@retry_on_exception()
def push_folder_to_hub(
    accelerator: Accelerator,
    output_dir: str,
    hf_repo_id: str | None = None,
    hf_repo_revision: str | None = None,
    private: bool = True,
):
    if accelerator.is_main_process:
        hf_repo_url = f"https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}"
        api = HfApi()
        if not api.repo_exists(hf_repo_id):
            api.create_repo(hf_repo_id, exist_ok=True, private=private)
        if hf_repo_revision is not None:
            api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
        api.upload_folder(
            repo_id=hf_repo_id,
            revision=hf_repo_revision,
            folder_path=output_dir,
            commit_message="upload checkpoint",
            run_as_future=False,
        )
        print(f"🔥 pushed to {hf_repo_url}")

- standardize evals (release evals?)

- when re-starting with long training stuff, will always restart from initial checkpoint again

allow max_steps to depend only on olmo config (just start once and let the olmo train until it ends).

- save init when training from scratch

it is not good that the environment variables that are passed to the training script are in the .sh file. should be in .yaml file

- add option to automatically upload final checkpoint to hf_hub (also for intermediate checkpoints)

add  (asynch?) evals during training.

- refactor num_steps per control and num_steps to work for the different szenatios	

- find a better option to decay / script change than restart with new yaml? although this is probably fine...

- handle config logic with better checks ... perhaps define a large args structure?

- allow for olmes - evals