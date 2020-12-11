#!/bin/sh
python -m scripts.train --config_filepath config/config.yaml > .out.nohup_sccnn_mnist_clfn_expt.out 2> .out.nohup_sccnn_mnist_clfn_error.out &
