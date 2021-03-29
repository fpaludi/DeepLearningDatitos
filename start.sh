tensorboard --logdir=src/lightning_logs --bind_all &
jupyter notebook . --ip 0.0.0.0 --no-browser --allow-root --port 4016
