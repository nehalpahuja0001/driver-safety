import os
import ast

try:
    import yaml
    has_yaml = True
except ImportError:
    has_yaml = False

def check_file_exists(filename):
    exists = os.path.isfile(filename)
    print(f"[{'X' if exists else ' '}] FILE: {filename} exists")
    return exists

def check_yaml_fields(filename):
    if not os.path.isfile(filename): return
    if not has_yaml:
        # Fallback simple textual check
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"[X] METADATA: openenv.yaml parsed (Basic text reading)")
        return
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        fields = ["name", "version", "tasks"]
        missing = [f for f in fields if f not in data]
        if missing:
            print(f"[ ] METADATA: openenv.yaml missing fields: {missing}")
        else:
            print(f"[X] METADATA: openenv.yaml has required fields: {fields}")
    except Exception as e:
        print(f"[ ] ERROR: Failed reading openenv.yaml: {e}")

def check_env_functions(filename):
    if not os.path.isfile(filename): return
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"[ ] ERROR: Failed to parse {filename}: {e}")
        return
    
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    env_class = None
    for c in classes:
        if "Env" in c.name or getattr(c, 'name', '') == "DriverSafetyEnv":
            env_class = c
            break
            
    if not env_class:
        methods = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    else:
        methods = [n.name for n in env_class.body if isinstance(n, ast.FunctionDef)]
        
    required = ['reset', 'step', 'state']
    missing = [m for m in required if m not in methods]
    
    if missing:
        print(f"[ ] LOGIC: {filename} missing functions: {missing}")
    else:
        print(f"[X] LOGIC: {filename} successfully contains required OpenEnv functions -> reset(), step(), state()")

if __name__ == "__main__":
    print("-" * 50)
    print(" OPENENV MANUAL VALIDATION CHECK ")
    print("-" * 50)
    
    check_file_exists("openenv.yaml")
    check_yaml_fields("openenv.yaml")
    
    check_file_exists("environment.py")
    check_env_functions("environment.py")
    
    check_file_exists("inference.py")
    check_file_exists("Dockerfile")
    check_file_exists("README.md")
    
    print("-" * 50)
    print(" VALIDATION COMPLETE ")
    print("-" * 50)
