# simplicial-2-complex-cnns
NeurIPS workshop paper: Simplicial 2-Complex Convolutional Neural Networks ; Authors: Eric Bunch, Qian You, Glenn Fung, Vikas Singh

Accepted at [Topological Data Analysis and Beyond: Workshop at NeurIPS 2020](https://tda-in-ml.github.io/).

arXiv entry [here](https://arxiv.org/abs/2012.06010).

## Setup
First configure the `config/config.yaml` file with `db_path`, `output_dir_path`, and `project_dir` appropriately. `db_path` should be the path to a `sqlite3` database that will contain metadata about experiments. `output_dir_path` should be a directory that will store the output files from each experiment. `project_dir` should be a directory that will be the destination for the preprocessed data.

Once this is configured, run `bash setup.sh`.

## Run
To run an experiment, configure the `config/config.yaml` file appropriately, and run `bash run.sh`.