#!/bin/sh
python -m scripts.train --config_filepath config/config.yaml > .out.std.out 2> .out.err.out &
