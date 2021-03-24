## hierarchical_nu docker

Here, we need to pass an SSH key in order to pip install the private GitHub repo. On MacOS, after creating an SSH key, run the following steps:

```
export DOCKER_BUILDKIT=1
eval `ssh-agent`
ssh-add ~/.ssh/my_key
```

Finally, to build the container, run:

```
docker build --ssh default=$SSH_AUTH_SOCK . -t hierarchical-nu
```

