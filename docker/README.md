## hierarchical_nu docker

Docker for running hierarchical_nu

### Building

Here, we need to pass an SSH key in order to pip install the private GitHub repo. On MacOS, after creating an SSH key, run the following steps:

```
export DOCKER_BUILDKIT=1
eval `ssh-agent`
ssh-add ~/.ssh/my_key
```

To build the container, run:

```
docker build --ssh default=$SSH_AUTH_SOCK . -t hierarchical-nu
```

### Running public image

There is an image available on Docker Hub a cescaralara/hierarchical-nu:latest. To grab this image run:

```
docker pull cescalara/hierarchical-nu:latest
```

See the docker docs for how to mount directories, launch jupyter notebooks etc.

To convert from docker hub to a singularity image:

```
singularity pull docker://cescalara/hierarchical-nu:latest
```


