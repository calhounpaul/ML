OPENDEVIN_WORKSPACE=$(pwd)/workspace
export WORKSPACE_BASE=$(pwd)/workspace
docker run -it --rm \
    --pull=always \
    -e SANDBOX_USER_ID=$(id -u) \
    -e PERSIST_SANDBOX="true" \
    -e SSH_PASSWORD="temp_pass" \
    -e WORKSPACE_MOUNT_PATH=$OPENDEVIN_WORKSPACE \
    -v $OPENDEVIN_WORKSPACE:/opt/workspace_base \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    -e LLM_API_KEY="ollama" \
    -e LLM_BASE_URL="http://host.docker.internal:11434" \
    --name opendevin-app-$(date +%Y%m%d%H%M%S) \
    ghcr.io/opendevin/opendevin:0.6

#command-r-plus:104b-q2_K