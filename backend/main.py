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
    allow_origins=[
        "http://localhost:4200",  # Angular default port
        "http://localhost:4223",  # Vite development server
        "http://127.0.0.1:4200", # Alternative localhost
        "http://127.0.0.1:4223", # Alternative localhost
        "*",  # Allow all for development/docker
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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
    """Register all service endpoints with proper async handling"""
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

                    import inspect
                    is_async_method = inspect.iscoroutinefunction(handler_method)

                    # Create endpoint based on method type and parameters
                    if methods == ["GET"] and not params:
                        # GET without parameters
                        if is_async_method:
                            async def async_get_endpoint():
                                return await handler_method()
                            endpoint_function = async_get_endpoint
                        else:
                            def sync_get_endpoint():
                                return handler_method()
                            endpoint_function = sync_get_endpoint

                    elif methods == ["POST"] and params:
                        from fastapi import Request
                        if params[0] == "training_data: dict":
                            # Training data upload
                            async def post_training_endpoint(request: Request):
                                data = await request.json()
                                if is_async_method:
                                    return await handler_method(data)
                                else:
                                    return handler_method(data)
                            endpoint_function = post_training_endpoint

                        elif "BackgroundTasks" in str(params):
                            # Background tasks
                            async def post_bg_endpoint(request: Request):
                                from fastapi import BackgroundTasks
                                tasks = BackgroundTasks()
                                data = await request.json() if request.method == "POST" else {}
                                if is_async_method:
                                    await handler_method(data, tasks)
                                else:
                                    handler_method(data, tasks)
                                return {"message": "Background task started"}
                            endpoint_function = post_bg_endpoint

                        elif params == ["file: UploadFile"]:
                            # File upload
                            async def post_file_endpoint(file):
                                if is_async_method:
                                    return await handler_method(file)
                                else:
                                    return handler_method(file)
                            endpoint_function = post_file_endpoint

                        else:
                            # Generic POST with data
                            async def post_generic_endpoint(request: Request):
                                data = await request.json()
                                if is_async_method:
                                    return await handler_method(data)
                                else:
                                    return handler_method(data)
                            endpoint_function = post_generic_endpoint

                    elif methods == ["GET"] and params:
                        # GET with path parameters
                        def get_with_params_endpoint(**kwargs):
                            if is_async_method:
                                import asyncio
                                loop = asyncio.new_event_loop()
                                try:
                                    return loop.run_until_complete(handler_method(**kwargs))
                                finally:
                                    loop.close()
                            else:
                                return handler_method(**kwargs)
                        endpoint_function = get_with_params_endpoint

                    else:
                        # Fallback - create async wrapper
                        if is_async_method:
                            async def async_fallback_endpoint():
                                return await handler_method()
                        else:
                            async def sync_fallback_endpoint():
                                return handler_method()
                        endpoint_function = async_fallback_endpoint if is_async_method else sync_fallback_endpoint

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
    print("🚀 Starting AI Training Backend...")
    print("📍 Host: 0.0.0.0")
    print("🔌 Port: 8000")
    print("🌐 CORS Origins: localhost:4200, localhost:4223, *")
    print("📋 Serving 31 endpoints for vision + speech services")
    uvicorn.run(app, host="0.0.0.0", port=8000)
