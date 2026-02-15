from configs.config import GUNICORN_WORKERS, GUNICORN_NUM_THREADS, GUNICORN_BACKLOG

reload = False

# logging
accesslog = '-'
loglevel = "INFO"

# process naming
proc_name = "ds-video-qc"

# server mechanics
# preload_app = False  # if set to true, cloud logging won't work for the workers
reuse_port = True

# server address
host = "0.0.0.0"
port = "8080"
bind = f"{host}:{port}"

# worker process settings
worker_class = "uvicorn.workers.UvicornWorker"
workers = GUNICORN_WORKERS
threads = GUNICORN_NUM_THREADS
BACKLOG = GUNICORN_BACKLOG
