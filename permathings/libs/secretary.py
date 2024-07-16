import os, sys, json, hashlib, getpass, subprocess

libs_dir = this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(libs_dir)
persistent_dir = os.path.dirname(libs_dir)
root_dir = os.path.dirname(persistent_dir)
ephemeral_dir = os.path.join(root_dir, 'ephemera')
secrets_file_path = os.path.join(ephemeral_dir, 'secrets.json')

DEFAULT_SECRETS = [
    'HF_TOKEN',
]

def get_current_hwid():
    hwid = None
    try:
        hwid = os.popen('hwid').read().strip().split(': ')[1]
    except Exception as e:
        print(f"Error getting HWID: {e}")
    return hwid

def get_hwid_sha3():
    HWID_SHA3 = hashlib.sha3_256(get_current_hwid().encode()).hexdigest()
    return HWID_SHA3

def get_secrets_if_exist():
    if not os.path.exists(secrets_file_path):
        return None
    else:
        with open(secrets_file_path, 'r') as f:
            secrets = json.load(f)
        return secrets
    
def validate_hwid():
    secrets = get_secrets_if_exist()
    if secrets is None:
        return False
    else:
        hwid_sha3_last3 = secrets.get('HWID_SHA3_LAST3')
        if hwid_sha3_last3 is None:
            return False
        else:
            if hwid_sha3_last3 == get_hwid_sha3()[-3:]:
                return True
            else:
                return False
            
def re_init_secrets_file():
    hwid_sha3 = get_hwid_sha3()
    secrets = {'HWID_SHA3_LAST3': hwid_sha3[-3:],}
    with open(secrets_file_path, 'w') as f:
        json.dump(secrets, f)
    return secrets

def encrypt_with_hwid(input_string):
    hwid = get_current_hwid()
    try:
        encrypted_string = subprocess.check_output(f'echo "{input_string}" | openssl enc -aes-256-cbc -a -salt -pass pass:{hwid} -pbkdf2', shell=True).decode().strip()
        return encrypted_string
    except subprocess.CalledProcessError as e:
        print(f"Error during encryption: {e}")
        return None

def decrypt_with_hwid(encrypted_string):
    hwid = get_current_hwid()
    try:
        decrypted_string = subprocess.check_output(f'echo "{encrypted_string}" | openssl enc -d -aes-256-cbc -a -salt -pass pass:{hwid} -pbkdf2', shell=True).decode().strip()
        return decrypted_string
    except subprocess.CalledProcessError as e:
        print(f"Error during decryption: {e}")
        return None

def set_secret(secret_name, secret_value):
    if validate_hwid():
        secrets_data = get_secrets_if_exist()
    else:
        secrets_data = re_init_secrets_file()
    secrets_data[secret_name] = encrypt_with_hwid(secret_value)
    with open(secrets_file_path, 'w') as f:
        json.dump(secrets_data, f)
    return True

def get_secret(secret_name):
    if validate_hwid():
        secrets_data = get_secrets_if_exist()
    else:
        secrets_data = re_init_secrets_file()
    secret = secrets_data.get(secret_name)
    if secret is None:
        secret = getpass.getpass(f"Enter {secret_name}: ")
        set_secret(secret_name, secret)
        return secret
    else:
        return decrypt_with_hwid(secret)

def get_all_defaults():
    for secret_name in DEFAULT_SECRETS:
        get_secret(secret_name)
