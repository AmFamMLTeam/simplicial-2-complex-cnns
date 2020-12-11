#!/bin/sh
python -m scripts.setup_db --config_filepath config/config.yaml > .out.std.out 2> .out.err.out &
