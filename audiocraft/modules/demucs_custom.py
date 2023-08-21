import torch

from demucs.pretrained import *
from demucs.pretrained import _parse_remote_files
from demucs.apply import Model
from demucs.states import load_model


class CustomRemoteRepo(RemoteRepo):
    def __init__(self, models: tp.Dict[str, str], cache_dir: str = None):
        super().__init__(models)

        self.cache_dir = cache_dir

    def get_model(self, sig: str) -> Model:
        try:
            url = self._models[sig]
        except KeyError:
            raise ModelLoadingError(
                f"Could not find a pre-trained model with signature {sig}."
            )
        pkg = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", check_hash=True, model_dir=self.cache_dir
        )  # type: ignore
        return load_model(pkg)


# modified: from demucs import pretrained
# self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(device)
def get_demucs_model(name: str, repo: tp.Optional[Path] = None, cache_dir: str = None):
    """`name` must be a bag of models name or a pretrained signature
    from the remote AWS model repo or the specified local repo if `repo` is not None.
    """
    if name == "demucs_unittest":
        return demucs_unittest()
    model_repo: ModelOnlyRepo
    if repo is None:
        models = _parse_remote_files(REMOTE_ROOT / "files.txt")
        model_repo = CustomRemoteRepo(models, cache_dir)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
    any_repo = AnyModelRepo(model_repo, bag_repo)
    model = any_repo.get_model(name)
    model.eval()
    return model
