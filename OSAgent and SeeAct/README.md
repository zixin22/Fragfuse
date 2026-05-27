# FragFuse

## Environment

Create and activate the Python environment:

```bash
conda create -n fragfuse python=3.9
conda activate fragfuse
```

Install the AGrail4Agent package:

```bash
cd AGrail4Agent
pip install -r requirements.txt
pip install -e .
```

## Docker

OSAgent runs commands inside a Docker container. Install Docker Desktop and verify it is available:

```bash
docker --version
```

Build the OSAgent image from the repository root:

```bash
docker build -t ubuntu AGrail4Agent/
```

Optional smoke test:

```bash
docker run -it ubuntu
```

## Data

The prepared FragFuse datasets are under:

```text
Datasets/OSAgent/
Datasets/SeeAct/
```

The default OSAgent memory bank is:

```text
AGrail4Agent/memory_bank-32.csv
```

## Commands

Run commands and dataset details are documented in:

```text
Datasets/OSAgent/README.md
Datasets/SeeAct/README.md
```
