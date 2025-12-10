"""
AI Training Backend - Minimal Main Application
Loads and initializes services directly
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sqlite3
import os

class TrainingRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 16
    val_split: float = 0.2
    use_lora: bool = False
    learning_rate: float = 0.001

# Headless setup
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/xdg-runtime'
os.environ['DISPLAY'] = ':0'

# Direct service imports
from speech_service import SpeechService
from vision_service import VisionService

# ===== SERVICE CONFIGURATION =====
def get_service_configurations():
    services = {}
    base_data_dir = Path("data")
    base_models_dir = Path("models")

    # Speech Service
    speech_config = SpeechService.get_service_config()
    speech_config.update({
        "class": SpeechService,
        "name": "speech",
        "init_params": {
            "models_dir": str(base_models_dir / "speech"),
            "data_dir": str(base_data_dir / "speech")
        }
    })
    services["speech"] = speech_config

    # Vision Service
    vision_config = VisionService.get_service_config()
    vision_config.update({
        "class": VisionService,
        "name": "vision",
        "init_params": {
            "models_dir": str(base_models_dir / "vision"),
            "data_dir": str(base_data_dir / "vision")
        }
    })
    services["vision"] = vision_config

    return services

# Get configurations
SERVICE_CONFIGS = get_service_configurations()

# ===== DATABASE INITIALIZATION =====
def initialize_databases(configs):
    """Initialize databases for all services"""
    Path("db").mkdir(exist_ok=True)

    for service_name, config in configs.items():
        db_name = f'db/{service_name}.db'

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        if 'database_schema' in config and 'tables' in config['database_schema']:
            for table_name, create_sql in config['database_schema']['tables'].items():
                try:
                    cursor.execute(create_sql.strip())
                    print(f"✓ Created table '{table_name}' in {service_name} database")
                except Exception as e:
                    print(f"Warning: Could not create table '{table_name}' in {service_name}: {e}")

        conn.commit()
        conn.close()
        print(f"✓ Database initialized for service: {service_name}")

# ===== FASTAPI APPLICATION =====
app = FastAPI(
    title="AI Training Backend",
    version="1.0.0",
    description="Minimal ML training backend supporting multiple services"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== SERVICE INITIALIZATION =====
def initialize_services(configs):
    """Initialize all services"""
    services = {}

    for service_name, config in configs.items():
        try:
            service_class = config['class']
            init_params = config.get('init_params', {})

            service_instance = service_class(**init_params)
            services[service_name] = service_instance
            print(f"✓ Service '{service_name}' initialized successfully")

        except Exception as e:
            print(f"✗ Failed to initialize service '{service_name}': {e}")
            raise

    return services

# Initialize databases and services
initialize_databases(SERVICE_CONFIGS)
services = initialize_services(SERVICE_CONFIGS)

# ===== ENDPOINT REGISTRATION =====
def register_endpoints(app, services, configs):
    """Register all service endpoints"""
    for service_name, service_instance in services.items():
        config = configs[service_name]

        # Health endpoint
        async def health_endpoint():
            return {"status": "healthy"}

        app.add_api_route(f"/{service_name}/health", health_endpoint, methods=["GET"])

        # Service-specific endpoints
        if 'endpoints' in config:
            for endpoint_config in config['endpoints']:
                try:
                    path = endpoint_config['path']
                    methods = endpoint_config['methods']
                    handler_name = endpoint_config['handler']

                    params = endpoint_config.get('params', [])

                    handler_method = getattr(service_instance, handler_name, None)
                    if not handler_method:
                        raise AttributeError(f"Handler method '{handler_name}' not found")

                    # Create a dynamic endpoint function that properly handles parameters
                    if params:
                        # Build parameter signature dynamically
                        param_names = []
                        param_imports = set()

                        for param in params:
                            param_parts = param.split(':')
                            param_name = param_parts[0].strip()
                            param_type = param_parts[1].strip() if len(param_parts) > 1 else 'str'
                            param_names.append(param_name)

                            # Track imports needed
                            if param_type == 'UploadFile':
                                param_imports.add('UploadFile')
                            elif param_type.startswith('BackgroundTasks'):
                                param_imports.add('BackgroundTasks')
                            elif param_type == 'TrainingRequest':
                                param_imports.add('TrainingRequest')

                        # Dynamically create the function - handle both sync and async methods
                        param_str = ', '.join(f'{name}: {param_type}' for name in param_names)
                        if handler_method.__name__.startswith('get_') or handler_method.__name__.endswith('_endpoint'):
                            # Sync methods - don't await
                            func_code = f"""
async def endpoint_function({param_str}):
    try:
        result = handler_method({', '.join(param_names)})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
                            """
                        else:
                            # Async methods - await
                            func_code = f"""
async def endpoint_function({param_str}):
    try:
        return await handler_method({', '.join(param_names)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
                            """

                        # Create the function with proper imports
                        namespace = {'handler_method': handler_method, 'HTTPException': HTTPException}
                        if 'UploadFile' in param_imports:
                            from fastapi import UploadFile
                            namespace['UploadFile'] = UploadFile
                        if 'BackgroundTasks' in param_imports:
                            from fastapi import BackgroundTasks
                            namespace['BackgroundTasks'] = BackgroundTasks
                        if 'TrainingRequest' in param_imports:
                            namespace['TrainingRequest'] = TrainingRequest

                        exec(func_code, namespace)
                        endpoint_function = namespace['endpoint_function']

                    else:
                        # No parameters - create unique wrapper for each endpoint
                        def create_endpoint_wrapper(handler):
                            async def endpoint_wrapper():
                                try:
                                    # Check if method is async or sync
                                    import inspect
                                    if inspect.iscoroutinefunction(handler):
                                        return await handler()
                                    else:
                                        return handler()
                                except Exception as e:
                                    raise HTTPException(status_code=500, detail=str(e))
                            return endpoint_wrapper

                        endpoint_function = create_endpoint_wrapper(handler_method)

                    app.add_api_route(path, endpoint_function, methods=methods, tags=[service_name])
                    print(f"✓ Registered endpoint: {methods[0]} {path}")

                except Exception as e:
                    print(f"✗ Failed to register endpoint {endpoint_config.get('path', 'unknown')}: {e}")

register_endpoints(app, services, SERVICE_CONFIGS)

# ===== GLOBAL ENDPOINTS =====
@app.get("/health")
async def health_check():
    """Global health check"""
    return {"status": "healthy", "services": list(services.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
