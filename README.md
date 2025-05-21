# Repositorio principal para el proyecto bps-az-hgsoc

## Setup
Para facilitar el cambio de máquinas se utilizará un arvhivo de definción de variables
de entorno `.env`. Dicho archivo se colocará en la raíz del repositorio, una vez clonado éste. Hay un fichero de ejemplo en los assets.

### Clonar el repositorio

Clonamos el repositorio, preferiblemente trabajar medinate `ssh`.
```
git clone git@gitlab.cbra.com:bps-az-hgsoc/hgsoc_ml.git
cd hgsoc_ml
```

### Definir variables de entorno específicas del usuario

Primero, copiamos el archivo `.env` de ejemplo
```
cp hgsoc_ml/resources/assets/.env .
```

Se asigna a la variable `ENV_FOLDER` la ruta donde se creará el entorno de conda. Supongamos que los entornos han de estar ubicados en, por ejemplo, `/NAS/users/lou/coda_envs`.

Además, se asigna a la variable `DATA_PATH` la ruta donde están los datos, por ejemplo, `/mnt/NAS/projects/bps_dumps/bps-az-hgsoc`. Por lo que el fichero `.env` quedaría algo así, finalmente:

```
ENV_FOLDER=/NAS/users/lou/coda_envs/hgsoc_ml
DATA_PATH=/mnt/NAS/projects/bps_dumps/bps-az-hgsoc

# -- Variables usadas en hgsoc_ml/omop/
# Ruta hacia los datos remototos
REMOTE_DATA_PATH="/folder_1/.../folder_n/data/"
# Usuario y nombre maquina (IP o según .ssh/config) a la que conectar
REMOTE_USER="username"
REMOTE_HOST="hostname"
# Ruta donde se guardarán localmente los datos
LOCAL_DATA_DIR="/folder_1/.../folder_n/data/"
# Ruta donde se guardan las tablas omop ampliadas
OMOP_EXTENDED_DIR="/folder_1/.../folder_n/OMOP_Vocab_Extended/"
```

Cabe señalar que la ruta `/NAS/users/lou/coda_envs/` debe existir y permitir la escritura.

### Instalación

Usaremos GNU `make` para simplificar la tarea de instalación, generación de binarios, documentación, etc. Por tanto, bastará con ejecutar el `makefile` del proyecto:

```
make
```

# Instalacion bps_to_omop como submodulo

Para poder hacer uso del código en el repositorio *bps_to_omop*, hemos decidido incluirlo en este mismo repositorio como [submodulo](https://git-scm.com/book/es/v2/Herramientas-de-Git-Subm%C3%B3dulos).

Esto nos permite desarrollar ambos proyectos a la vez, adaptando *bps_to_omop* **en una rama específica** según nos sea necesario.

Para incluir *bps_to_omop* como submodulo creamos una nueva carpeta `lib/` y usamos el comando submodule de git:

    mkdir lib
    git submodule add https://gitlab.clinbioinfosspa.es/igutierrez/bps_to_omop.git lib/bps_to_omop
    git commit -m "Add bps_to_omop as a submodule"

    cd lib/bps_to_omop
    git branch hgsoc
    git checkout hgsoc

Es importante destacar que si hacemos cualquier cambio en *bps_to_omop* dentro de `lib/` **será necesario hacer el commit para ambos repositorios**, primero en *bps_to_omop* y luego en *hgsoc*. 