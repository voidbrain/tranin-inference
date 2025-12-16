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
    """Register all service endpoints with proper service isolation"""
    global_endpoints = []
    service_endpoints = {}

    # Separate endpoints by service
    for service_name, config in configs.items():
        if 'endpoints' in config:
            for endpoint_config in config['endpoints']:
                path = endpoint_config['path']
                if path.startswith('/'):
                    # Global endpoints (shared across services)
                    global_endpoints.append((endpoint_config, service_name))
                else:
                    # Service-specific endpoints
                    if service_name not in service_endpoints:
                        service_endpoints[service_name] = []
                    service_endpoints[service_name].append((endpoint_config, service_name))

    # Register health endpoints for each service
    for service_name in services.keys():
        async def health_endpoint(service=service_name):
            return {"status": "healthy", "service": service}

        app.add_api_route(f"/{service_name}/health", health_endpoint, methods=["GET"], tags=[service_name])

    # Register global endpoints
    for endpoint_config, service_name in global_endpoints:
        try:
            register_single_endpoint(app, endpoint_config, services[service_name], service_name)
        except Exception as e:
            print(f"✗ Failed to register global endpoint {endpoint_config.get('path', 'unknown')}: {e}")

    # Register service-specific endpoints
    for service_name, endpoints in service_endpoints.items():
        for endpoint_config, _ in endpoints:
            try:
                register_single_endpoint(app, endpoint_config, services[service_name], service_name)
            except Exception as e:
                print(f"✗ Failed to register {service_name} endpoint {endpoint_config.get('path', 'unknown')}: {e}")

def register_single_endpoint(app, endpoint_config, service_instance, service_name):
    """Register a single endpoint with proper error handling"""
    path = endpoint_config['path']
    methods = endpoint_config['methods']
    handler_name = endpoint_config['handler']
    params = endpoint_config.get('params', [])

    handler_method = getattr(service_instance, handler_name, None)
    if not handler_method:
        raise AttributeError(f"Handler method '{handler_name}' not found in {service_name} service")

    import inspect
    is_async_method = inspect.iscoroutinefunction(handler_method)

    # Create proper endpoint function
    if methods == ["GET"] and not params:
        # Simple GET without parameters
        async def get_endpoint():
            try:
                if is_async_method:
                    return await handler_method()
                else:
                    import asyncio
                    import concurrent.futures
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        return await loop.run_in_executor(executor, handler_method)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Endpoint error: {str(e)}")

        endpoint_function = get_endpoint

    elif methods == ["POST"] and params:
        from fastapi import Request
        if params == ["training_data: dict", "background_tasks: BackgroundTasks"]:
            # Special case: training data dict + background tasks dependency injection
            from fastapi import BackgroundTasks

            async def post_training_with_background(training_data: dict, background_tasks: BackgroundTasks):
                try:
                    if is_async_method:
                        return await handler_method(training_data, background_tasks)
                    else:
                        import asyncio
                        import concurrent.futures
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            return await loop.run_in_executor(executor, handler_method, training_data, background_tasks)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Training data error: {str(e)}")

            endpoint_function = post_training_with_background

        elif params[0] == "training_data: dict":
            async def post_training_endpoint(request: Request):
                try:
                    data = await request.json()
                    if is_async_method:
                        return await handler_method(data)
                    else:
                        import asyncio
                        import concurrent.futures
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            return await loop.run_in_executor(executor, handler_method, data)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Training data error: {str(e)}")

            endpoint_function = post_training_endpoint

        elif "UploadFile" in str(params):
            # Handle file upload endpoints (UploadFile parameters)
            from fastapi import UploadFile, File, Query

            # Get all parameter info from the method signature
            import inspect
            sig = inspect.signature(handler_method)
            method_params = list(sig.parameters.keys())
            # Skip 'self' parameter
            method_params = [p for p in method_params if p != 'self']

            # Separate UploadFile params from other params (which should be query params)
            upload_params = []
            query_params = []

            for param_spec in params:
                param_name = param_spec.split(':')[0]
                if 'UploadFile' in param_spec:
                    upload_params.append(param_name)
                else:
                    query_params.append(param_name)

            # Create dynamic function based on parameter types
            if len(upload_params) == 1 and len(query_params) == 1:
                # Special case: one file upload + one query parameter
                upload_param = upload_params[0]
                query_param = query_params[0]

                async def post_mixed_endpoint(audio_file: UploadFile = File(...), language: str = Query(...)):
                    try:
                        if is_async_method:
                            return await handler_method(audio_file, language)
                        else:
                            import asyncio
                            import concurrent.futures
                            loop = asyncio.get_event_loop()
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                return await loop.run_in_executor(executor, handler_method, audio_file, language)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

                endpoint_function = post_mixed_endpoint

            elif len(upload_params) == 1 and len(query_params) == 2:
                # Special case: one file upload + two query parameters
                upload_param = upload_params[0]
                query_param1 = query_params[0]
                query_param2 = query_params[1]

                async def post_mixed_endpoint_2qp(audio_file: UploadFile = File(...), language: str = Query(...), transcript: str = Query(...)):
                    try:
                        if is_async_method:
                            return await handler_method(audio_file, language, transcript)
                        else:
                            import asyncio
                            import concurrent.futures
                            loop = asyncio.get_event_loop()
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                return await loop.run_in_executor(executor, handler_method, audio_file, language, transcript)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

                endpoint_function = post_mixed_endpoint_2qp

            elif len(upload_params) == 1 and len(query_params) == 0:
                # Standard file upload only
                param_name = upload_params[0]

                async def post_file_upload_endpoint(audio_file: UploadFile = File(...)):
                    try:
                        if is_async_method:
                            return await handler_method(audio_file)
                        else:
                            import asyncio
                            import concurrent.futures
                            loop = asyncio.get_event_loop()
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                return await loop.run_in_executor(executor, handler_method, audio_file)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

                endpoint_function = post_file_upload_endpoint
            else:
                # Fallback for other cases
                async def post_file_upload_fallback(file: UploadFile = File(...)):
                    try:
                        if is_async_method:
                            return await handler_method(file)
                        else:
                            import asyncio
                            import concurrent.futures
                            loop = asyncio.get_event_loop()
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                return await loop.run_in_executor(executor, handler_method, file)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

                endpoint_function = post_file_upload_fallback

        elif params == ["training_data: dict", "background_tasks: BackgroundTasks"]:
            # Special case: training data dict + background tasks
            async def post_training_with_background(training_data: dict, background_tasks: "BackgroundTasks"):
                try:
                    if is_async_method:
                        return await handler_method(training_data, background_tasks)
                    else:
                        import asyncio
                        import concurrent.futures
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            return await loop.run_in_executor(executor, handler_method, training_data, background_tasks)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Training data error: {str(e)}")

            endpoint_function = post_training_with_background

        else:
            # Handle generic POST endpoints with parameters
            async def post_generic_endpoint(request: Request):
                try:
                    data = await request.json()

                    # Extract parameter values from the JSON data based on the method signature
                    import inspect
                    sig = inspect.signature(handler_method)
                    method_params = list(sig.parameters.keys())

                    # Skip 'self' parameter
                    method_params = [p for p in method_params if p != 'self']

                    # Prepare arguments
                    args = []
                    kwargs = {}

                    # If there's only one parameter and it's named the same as what was sent, extract it
                    if len(method_params) == 1 and len(params) == 1:
                        param_name = params[0].split(':')[0]  # e.g., "language" from "language: str"
                        if param_name in data:
                            args.append(data[param_name])
                        else:
                            # If the data is already the parameter value itself
                            args.append(data)
                    else:
                        # For multiple parameters or unknown cases, pass as kwargs
                        for param in params:
                            param_name = param.split(':')[0]
                            if param_name in data:
                                kwargs[param_name] = data[param_name]

                    # Call the handler method
                    if is_async_method:
                        if args:
                            return await handler_method(*args)
                        else:
                            return await handler_method(**kwargs)
                    else:
                        import asyncio
                        import concurrent.futures
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            if args:
                                return await loop.run_in_executor(executor, handler_method, *args)
                            else:
                                return await loop.run_in_executor(executor, handler_method, **kwargs)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"POST error: {str(e)}")

            endpoint_function = post_generic_endpoint

    elif methods == ["GET"] and params:
        # Check if this endpoint has path parameters (contains {param} in path)
        import re
        path_params = re.findall(r'\{([^}]+)\}', path)

        if path_params:
            # Create function signature with exact path parameter names
            param_names = [p.split(':')[0] for p in params]  # Remove type annotations

            if len(path_params) == 1 and len(param_names) == 1:
                # Single path parameter case - create a function with correct signature
                param_name = path_params[0]

                # Create a function that properly expects the path parameter
                # Use dynamic function creation to handle different parameter names
                if param_name == 'detection_mode':
                    # Special case for detection_mode path parameter
                    async def path_endpoint_detection_mode(detection_mode: str):
                        try:
                            if is_async_method:
                                return await handler_method(detection_mode)
                            else:
                                import asyncio
                                import concurrent.futures
                                loop = asyncio.get_event_loop()
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    return await loop.run_in_executor(executor, handler_method, detection_mode)
                        except Exception as e:
                            raise HTTPException(status_code=500, detail=f"GET path param error: {str(e)}")

                    endpoint_function = path_endpoint_detection_mode
                elif param_name == 'training_type':
                    # Special case for training_type path parameter
                    async def path_endpoint_training_type(training_type: str):
                        try:
                            if is_async_method:
                                return await handler_method(training_type)
                            else:
                                import asyncio
                                import concurrent.futures
                                loop = asyncio.get_event_loop()
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    return await loop.run_in_executor(executor, handler_method, training_type)
                        except Exception as e:
                            raise HTTPException(status_code=500, detail=f"GET path param error: {str(e)}")

                    endpoint_function = path_endpoint_training_type
                else:
                    # Generic fallback for other path parameters
                    async def generic_path_endpoint(param_value: str):
                        try:
                            import asyncio
                            if is_async_method:
                                return await handler_method(param_value)
                            else:
                                import concurrent.futures
                                loop = asyncio.get_event_loop()
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    return await loop.run_in_executor(executor, handler_method, param_value)
                        except Exception as e:
                            raise HTTPException(status_code=500, detail=f"GET generic path param error: {str(e)}")

                    endpoint_function = generic_path_endpoint
            else:
                # Multiple parameters - fallback to kwargs
                async def get_mixed_params_endpoint(**kwargs):
                    try:
                        if is_async_method:
                            return await handler_method(**kwargs)
                        else:
                            import asyncio
                            import concurrent.futures
                            loop = asyncio.get_event_loop()
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                return await loop.run_in_executor(executor, handler_method, **kwargs)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"GET mixed params error: {str(e)}")

                endpoint_function = get_mixed_params_endpoint
        else:
            # Query parameters only
            async def get_query_params_endpoint(**kwargs):
                try:
                    if is_async_method:
                        return await handler_method(**kwargs)
                    else:
                        import asyncio
                        import concurrent.futures
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            return await loop.run_in_executor(executor, handler_method, **kwargs)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"GET query params error: {str(e)}")

            endpoint_function = get_query_params_endpoint

    else:
        # Default async wrapper
        async def default_endpoint():
            try:
                if is_async_method:
                    return await handler_method()
                else:
                    import asyncio
                    import concurrent.futures
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        return await loop.run_in_executor(executor, handler_method)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Default endpoint error: {str(e)}")

        endpoint_function = default_endpoint

    app.add_api_route(path, endpoint_function, methods=methods, tags=[service_name])
    print(f"✓ Registered endpoint: {methods[0]} {path} [{service_name}]")

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
