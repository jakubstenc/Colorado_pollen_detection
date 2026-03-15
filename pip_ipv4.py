import socket
import sys
import subprocess

# Monkeypatch socket to force IPv4
orig_getaddrinfo = socket.getaddrinfo

def filtered_getaddrinfo(*args, **kwargs):
    # If family is not specified or is AF_UNSPEC, force AF_INET
    family = kwargs.get('family', socket.AF_UNSPEC)
    if family == socket.AF_UNSPEC:
        kwargs['family'] = socket.AF_INET
    
    try:
        results = orig_getaddrinfo(*args, **kwargs)
        # Verify it's IPv4
        return [r for r in results if r[0] == socket.AF_INET]
    except socket.gaierror:
        # Fallback to standard if forcing fails, but we prefer AF_INET
        return orig_getaddrinfo(*args, **kwargs)

socket.getaddrinfo = filtered_getaddrinfo

if __name__ == "__main__":
    # Run pip as a module within the interpreter that has the patch
    # We must ensure we use the pip from the current environment (venv)
    sys.exit(subprocess.call([sys.executable, "-m", "pip"] + sys.argv[1:]))
