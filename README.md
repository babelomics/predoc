# Main repository for the predOC project

## Setup
To facilitate switching between machines, an environment variable definition file `.env` will be used. This file will be placed in the root of the repository once it is cloned. There is an example file in the assets.

### Clone the repository

Clone the repository, preferably working via `ssh`.
```
git@github.com:babelomics/predoc.git
cd predoc
```

### Define user-specific environment variables

First, copy the example `.env` file.
```
cp predoc/resources/assets/.env .
```

Assign the path where the conda environment will be created to the `ENV_FOLDER` variable. Let's assume the environments should be located in, for example, `~/conda_envs`.

Additionally, assign the path where the data is located to the `DATA_PATH` variable, for example, `~/projects/bps_dumps/predoc`. Therefore, the `.env` file would finally look something like this:

```
ENV_FOLDER=~/conda_envs
DATA_PATH=~/projects/bps_dumps/predoc

# -- Variables used in predoc/omop/
# Path to remote data
REMOTE_DATA_PATH="/folder_1/.../folder_n/data/"
# User and machine name (IP or according to .ssh/config) to connect to
REMOTE_USER="username"
REMOTE_HOST="hostname"
# Path where data will be saved locally
LOCAL_DATA_DIR="/folder_1/.../folder_n/data/"
# Path where extended OMOP tables are saved
OMOP_EXTENDED_DIR="/folder_1/.../folder_n/OMOP_Vocab_Extended/"
```

It should be noted that the path `~/conda_envs` must exist and allow writing.

### Installation

We will use GNU `make` to simplify the task of installation, generation of binaries, documentation, etc. Therefore, it will be sufficient to execute the project's `makefile`:

```
make
```

# Installation of bps_to_omop as a submodule

To be able to use the code in the *bps_to_omop* repository, we have decided to include it in this same repository as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

This allows us to develop both projects at the same time, adapting *bps_to_omop* **on a specific branch** as needed.

To include *bps_to_omop* as a submodule, we create a new `lib/` folder and use the git submodule command:

```
mkdir lib
git submodule add [https://gitlab.clinbioinfosspa.es/igutierrez/bps_to_omop.git](https://gitlab.clinbioinfosspa.es/igutierrez/bps_to_omop.git) lib/bps_to_omop
git commit -m "Add bps_to_omop as a submodule"

cd lib/bps_to_omop
git branch hgsoc
git checkout hgsoc
```

It is important to note that if we make any changes to *bps_to_omop* within `lib/` **it will be necessary to commit for both repositories**, first in *bps_to_omop* and then in *hgsoc*.
