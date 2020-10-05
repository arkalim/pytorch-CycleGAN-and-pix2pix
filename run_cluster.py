import os

os.system("polyaxon run -f polyaxonfile.yaml -u -l")

# polyaxon tensorboard -xp 15545 start -f tensorboard.yaml