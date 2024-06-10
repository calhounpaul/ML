import os, sys, json, re, time, datetime, random, string, docker, atexit

this_modules_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(this_modules_path)
sys.path.append(parent_dir)

def run_docker_container(image_name, port_bindings, environment_vars, volumes, detach=True):
    client = docker.from_env()
    container = client.containers.run(image_name, detach=detach, ports=port_bindings, environment=environment_vars, volumes=volumes)
    return container

def create_docker_image_from_dockerfile(dockerfile_path, image_name, tag="latest"):
    client = docker.from_env()
    image, build_log = client.images.build(path=dockerfile_path, tag=f"{image_name}:{tag}")
    return image, build_log