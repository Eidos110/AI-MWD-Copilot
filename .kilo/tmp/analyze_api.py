"""Decode the carrier instance keyed_id from the API JSON."""
import json

# From the API response I got:
data = {
    "id": "4da51564-3203-4663-9ec3-810946629d77",
    "serviceId": "25d24257-bda3-4fa0-a6ab-4f797107da8e",
    "serviceName": "mwd-backend",
    "domains": {
        "serviceDomains": [
            {"id": "2b0a179f-28aa-40c4-87ba-1aaf083455be",
             "domain": "mwd-backend-production.up.railway.app",
             "targetPort": None}
        ],
        "customDomains": []
    },
    "latestDeployment": {
        "meta": {
            "build": {
                "builder": "DOCKERFILE",
                "dockerfilePath": "/backend/Dockerfile",
                "buildCommand": "pip install --no-cache-dir -r requirements.txt",
                "rootDirectory": "backend"
            },
            "deploy": {
                "healthcheckPath": "/",
                "healthcheckTimeout": 600,
                "runningPort": null,
                "startCommand": "uvicorn backend.app:app --host 0.0.0.0 --port 8000"
            }
        },
        "status": "FAILED"
    }
}

print("=== Carrier Instance Analysis ===")
print(f"Service instance: {data['id']}")
print(f"Domain: {data['domains']['serviceDomains'][0]['domain']}")
build = data['latestDeployment']['meta']['build']
deploy = data['latestDeployment']['meta']['deploy']
print(f"Builder: {build['builder']}")
print(f"Dockerfile path: {build['dockerfilePath']}")
print(f"Build command: {build['buildCommand']}")
print(f"Root directory: {build['rootDirectory']}")
print(f"Start command: {deploy['startCommand']}")
print(f"Healthcheck path: {deploy['healthcheckPath']}")
print(f"Healthcheck timeout: {deploy['healthcheckTimeout']}s")
print(f"Running port: {deploy['runningPort']}")
print(f"Latest deployment status: {data['latestDeployment']['status']}")
print()

# WHAT I NEED TO FIX:
print("=== Issues to Fix ===")
print(f"1. Healthcheck path is '/' - should be '/health' for the carrier to use")
print(f"2. RunningPort is null - should be 8000 (Uvicorn default in start.sh)")
print(f"3. Dockerfile path is '/backend/Dockerfile' - verifies Dockerfile is at backend/Dockerfile relative to context root")
print(f"4. Build command is '{build['buildCommand']}' - but our Dockerfile has its own RUN pip install")
print()

# From the service manifest:
# dockerfilePath: "/backend/Dockerfile" -- this means: look for Dockerfile in /backend/ within context root
# But rootDirectory is "backend" -- so context root is the backend/ folder
# If context root is backend/ and dockerfilePath is "/backend/Dockerfile":
#   → Looking for /backend/Dockerfile inside the backend/ context = looking for backend/backend/Dockerfile (WRONG)
# Unless the / in dockerfilePath means absolute path = root of context = backend/Dockerfile (OK)
print("Interpretation:")
print("rootDirectory='backend' → context root = backend/ folder")
print("dockerfilePath='/backend/Dockerfile' → absolute from context root = backend/Dockerfile")
print("START COMMAND: uvicorn backend.app:app --host 0.0.0.0 --port 8000")
