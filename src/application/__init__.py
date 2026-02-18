from importlib import metadata

__app_name__ = metadata.metadata("ds-video-qc")["Name"]
__version__ = metadata.version("ds-video-qc")