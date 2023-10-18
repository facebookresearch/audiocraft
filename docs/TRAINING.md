# AudioCraft training pipelines

AudioCraft training pipelines are built on top of PyTorch as our core deep learning library
and [Flashy](https://github.com/facebookresearch/flashy) as our training pipeline design library,
and [Dora](https://github.com/facebookresearch/dora) as our experiment manager.
AudioCraft training pipelines are designed to be research and experiment-friendly.


## Environment setup

For the base installation, follow the instructions from the [README.md](../README.md).
Below are some additional instructions for setting up the environment to train new models.

### Team and cluster configuration

In order to support multiple teams and clusters, AudioCraft uses an environment configuration.
The team configuration allows to specify cluster-specific configurations (e.g. SLURM configuration),
or convenient mapping of paths between the supported environments.

Each team can have a yaml file under the [configuration folder](../config). To select a team set the
`AUDIOCRAFT_TEAM` environment variable to a valid team name (e.g. `labs` or `default`):
```shell
conda env config vars set AUDIOCRAFT_TEAM=default
```

Alternatively, you can add it to your `.bashrc`:
```shell
export AUDIOCRAFT_TEAM=default
```

If not defined, the environment will default to the `default` team.

The cluster is automatically detected, but it is also possible to override it by setting
the `AUDIOCRAFT_CLUSTER` environment variable.

Based on this team and cluster, the environment is then configured with:
* The dora experiment outputs directory.
* The available slurm partitions: categorized by global and team.
* A shared reference directory: In order to facilitate sharing research models while remaining
agnostic to the used compute cluster, we created the `//reference` symbol that can be used in
YAML config to point to a defined reference folder containing shared checkpoints
(e.g. baselines, models for evaluation...).

**Important:** The default output dir for trained models and checkpoints is under `/tmp/`. This is suitable
only for quick testing. If you are doing anything serious you MUST edit the file `default.yaml` and
properly set the `dora_dir` entries.

#### Overriding environment configurations

You can set the following environment variables to bypass the team's environment configuration:
* `AUDIOCRAFT_CONFIG`: absolute path to a team config yaml file.
* `AUDIOCRAFT_DORA_DIR`: absolute path to a custom dora directory.
* `AUDIOCRAFT_REFERENCE_DIR`: absolute path to the shared reference directory.

## Training pipelines

Each task supported in AudioCraft has its own training pipeline and dedicated solver.
Learn more about solvers and key designs around AudioCraft training pipeline below.
Please refer to the documentation of each task and model for specific information on a given task.


### Solvers

The core training component in AudioCraft is the solver. A solver holds the definition
of how to solve a given task: It implements the training pipeline logic, combining the datasets,
model, optimization criterion and components and the full training loop. We refer the reader
to [Flashy](https://github.com/facebookresearch/flashy) for core principles around solvers.

AudioCraft proposes an initial solver, the `StandardSolver` that is used as the base implementation
for downstream solvers. This standard solver provides a nice base management of logging,
checkpoints loading/saving, xp restoration, etc. on top of the base Flashy implementation.
In AudioCraft, we made the assumption that all tasks are following the same set of stages:
train, valid, evaluate and generation, each relying on a dedicated dataset.

Each solver is responsible for defining the task to solve and the associated stages
of the training loop in order to leave the full ownership of the training pipeline
to the researchers. This includes loading the datasets, building the model and
optimisation components, registering them and defining the execution of each stage.
To create a new solver for a given task, one should extend the StandardSolver
and define each stage of the training loop. One can further customise its own solver
starting from scratch instead of inheriting from the standard solver.

```python
from . import base
from .. import optim


class MyNewSolver(base.StandardSolver):

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        # one can add custom attributes to the solver
        self.criterion = torch.nn.L1Loss()

    def best_metric(self):
        # here optionally specify which metric to use to keep track of best state
        return 'loss'

    def build_model(self):
        # here you can instantiate your models and optimization related objects
        # this method will be called by the StandardSolver init method
        self.model = ...
        # the self.cfg attribute contains the raw configuration
        self.optimizer = optim.build_optimizer(self.model.parameters(), self.cfg.optim)
        # don't forget to register the states you'd like to include in your checkpoints!
        self.register_stateful('model', 'optimizer')
        # keep the model best state based on the best value achieved at validation for the given best_metric
        self.register_best('model')
        # if you want to add EMA around the model
        self.register_ema('model')

    def build_dataloaders(self):
        # here you can instantiate your dataloaders
        # this method will be called by the StandardSolver init method
        self.dataloaders = ...

    ...

    # For both train and valid stages, the StandardSolver relies on
    # a share common_train_valid implementation that is in charge of
    # accessing the appropriate loader, iterate over the data up to
    # the specified number of updates_per_epoch, run the ``run_step``
    # function that you need to implement to specify the behavior
    # and finally update the EMA and collect the metrics properly.
    @abstractmethod
    def run_step(self, idx: int, batch: tp.Any, metrics: dict):
        """Perform one training or valid step on a given batch.
        """
        ... # provide your implementation of the solver over a batch

    def train(self):
        """Train stage.
        """
        return self.common_train_valid('train')

    def valid(self):
        """Valid stage.
        """
        return self.common_train_valid('valid')

    @abstractmethod
    def evaluate(self):
        """Evaluate stage.
        """
        ... # provide your implementation here!

    @abstractmethod
    def generate(self):
        """Generate stage.
        """
        ... # provide your implementation here!
```

### About Epochs

AudioCraft Solvers uses the concept of Epoch. One epoch doesn't necessarily mean one pass over the entire
dataset, but instead represent the smallest amount of computation that we want to work with before checkpointing.
Typically, we find that having an Epoch time around 30min is ideal both in terms of safety (checkpointing often enough)
and getting updates often enough. One Epoch is at least a `train` stage that lasts for `optim.updates_per_epoch` (2000 by default),
and a `valid` stage. You can control how long the valid stage takes with `dataset.valid.num_samples`.
Other stages (`evaluate`, `generate`) will only happen every X epochs, as given by `evaluate.every` and `generate.every`).


### Models

In AudioCraft, a model is a container object that wraps one or more torch modules together
with potential processing logic to use in a solver. For example, a model would wrap an encoder module,
a quantisation bottleneck module, a decoder and some tensor processing logic. Each of the previous components
can be considered as a small « model unit » on its own but the container model is a practical component
to manipulate and train a set of modules together.

### Datasets

See the [dedicated documentation on datasets](./DATASETS.md).

### Metrics

See the [dedicated documentation on metrics](./METRICS.md).

### Conditioners

AudioCraft language models can be conditioned in various ways and the codebase offers a modular implementation
of different conditioners that can be potentially combined together.
Learn more in the [dedicated documentation on conditioning](./CONDITIONING.md).

### Configuration

AudioCraft's configuration is defined in yaml files and the framework relies on
[hydra](https://hydra.cc/docs/intro/) and [omegaconf](https://omegaconf.readthedocs.io/) to parse
and manipulate the configuration through Dora.

##### :warning: Important considerations around configurations

Our configuration management relies on Hydra and the concept of group configs to structure
and compose configurations. Updating the root default configuration files will then have
an impact on all solvers and tasks.
**One should never change the default configuration files. Instead they should use Hydra config groups in order to store custom configuration.**
Once this configuration is created and used for running experiments, you should not edit it anymore.

Note that as we are using Dora as our experiment manager, all our experiment tracking is based on
signatures computed from delta between configurations.
**One must therefore ensure backward compatibility of the configuration at all time.**
See [Dora's README](https://github.com/facebookresearch/dora) and the
[section below introduction Dora](#running-experiments-with-dora).

##### Configuration structure

The configuration is organized in config groups:
* `conditioner`: default values for conditioning modules.
* `dset`: contains all data source related information (paths to manifest files
and metadata for a given dataset).
* `model`: contains configuration for each model defined in AudioCraft and configurations
for different variants of models.
* `solver`: contains the default configuration for each solver as well as configuration
for each solver task, combining all the above components.
* `teams`: contains the cluster configuration per teams. See environment setup for more details.

The `config.yaml` file is the main configuration that composes the above groups
and contains default configuration for AudioCraft.

##### Solver's core configuration structure

The core configuration structure shared across solver is available in `solvers/default.yaml`.

##### Other configuration modules

AudioCraft configuration contains the different setups we used for our research and publications.

## Running experiments with Dora

### Launching jobs

Try launching jobs for different tasks locally with dora run:

```shell
# run compression task with lightweight encodec
dora run solver=compression/debug
```

Most of the time, the jobs are launched through dora grids, for example:

```shell
# run compression task through debug grid
dora grid compression.debug
```

Learn more about running experiments with Dora below.

### A small introduction to Dora

[Dora](https://github.com/facebookresearch/dora) is the experiment manager tool used in AudioCraft.
Check out the README to learn how Dora works. Here is a quick summary of what to know:
* An XP is a unique set of hyper-parameters with a given signature. The signature is a hash
of those hyper-parameters. We always refer to an XP with its signature, e.g. 9357e12e. We will see
after that one can retrieve the hyper-params and re-rerun it in a single command.
* In fact, the hash is defined as a delta between the base config and the one obtained
with the config overrides you passed from the command line. This means you must never change
the `conf/**.yaml` files directly, except for editing things like paths. Changing the default values
in the config files means the XP signature won't reflect that change, and wrong checkpoints might be reused.
I know, this is annoying, but the reason is that otherwise, any change to the config file would mean
that all XPs ran so far would see their signature change.

#### Dora commands

```shell
dora info -f 81de367c  # this will show the hyper-parameter used by a specific XP.
                       # Be careful some overrides might present twice, and the right most one
                       # will give you the right value for it.

dora run -d -f 81de367c   # run an XP with the hyper-parameters from XP 81de367c.
                          # `-d` is for distributed, it will use all available GPUs.

dora run -d -f 81de367c dataset.batch_size=32  # start from the config of XP 81de367c but change some hyper-params.
                                               # This will give you a new XP with a new signature (e.g. 3fe9c332).

dora info -f SIG -t    # will tail the log (if the XP has scheduled).
# if you need to access the logs of the process for rank > 0, in particular because a crash didn't happen in the main
# process, then use `dora info -f SIG` to get the main log name (finished into something like `/5037674_0_0_log.out`)
# and worker K can be accessed as `/5037674_0_{K}_log.out`.
# This is only for scheduled jobs, for local distributed runs with `-d`, then you should go into the XP folder,
# and look for `worker_{K}.log` logs.
```

An XP runs from a specific folder based on its signature, under the
`<cluster_specific_path>/<user>/experiments/audiocraft/outputs/` folder.
You can safely interrupt a training and resume it, it will reuse any existing checkpoint,
as it will reuse the same folder. If you made some change to the code and need to ignore
a previous checkpoint you can use `dora run --clear [RUN ARGS]`.

If you have a Slurm cluster, you can also use the dora grid command, e.g.

```shell
# Run a dummy grid located at `audiocraft/grids/my_grid_folder/my_grid_name.py`
dora grid my_grid_folder.my_grid_name
# The following will simply display the grid and also initialize the Dora experiments database.
# You can then simply refer to a config using its signature (e.g. as `dora run -f SIG`).
dora grid my_grid_folder.my_grid_name --dry_run --init
```

Please refer to the [Dora documentation](https://github.com/facebookresearch/dora) for more information.


#### Clearing up past experiments

```shell
# This will cancel all the XPs and delete their folder and checkpoints.
# It will then reschedule them starting from scratch.
dora grid my_grid_folder.my_grid_name --clear
# The following will delete the folder and checkpoint for a single XP,
# and then run it afresh.
dora run [-f BASE_SIG] [ARGS] --clear
```
