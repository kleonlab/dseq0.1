#tyhis is the savanna version. 

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arcinstitute/savanna_evo2_1b_base",
    local_dir="models/evo",
    local_dir_use_symlinks=False
)