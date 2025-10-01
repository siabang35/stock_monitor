#!/usr/bin/env python3
"""
Documentation Generator for Warehouse Stock Counting System

This script automatically generates comprehensive documentation from the codebase,
including API documentation, system architecture, and usage guides.
"""

import os
import json
import ast
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DocumentationGenerator:
    """Generate comprehensive documentation from codebase"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.docs_dir.mkdir(exist_ok=True)
        
        # Documentation structure
        self.api_docs = {
            "title": "Warehouse Stock Counting API",
            "version": "1.0.0",
            "description": "REST API for warehouse stock monitoring and counting",
            "endpoints": [],
            "models": [],
            "websockets": []
        }
        
        self.system_docs = {
            "title": "System Architecture",
            "components": {},
            "dependencies": {},
            "file_structure": {},
            "configuration": {}
        }
        
        self.usage_docs = {
            "title": "Usage Guide",
            "installation": [],
            "configuration": [],
            "examples": [],
            "troubleshooting": []
        }
    
    def extract_fastapi_endpoints(self, file_path: str) -> List[Dict]:
        """Extract FastAPI endpoints from main.py"""
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for FastAPI decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr'):
                            method = decorator.func.attr.lower()
                            if method in ['get', 'post', 'put', 'delete', 'patch']:
                                # Extract endpoint info
                                endpoint = {
                                    "method": method.upper(),
                                    "path": "",
                                    "function": node.name,
                                    "description": "",
                                    "parameters": [],
                                    "returns": {}
                                }
                                
                                # Get path from decorator
                                if decorator.args:
                                    if isinstance(decorator.args[0], ast.Constant):
                                        endpoint["path"] = decorator.args[0].value
                                
                                # Get docstring
                                if (node.body and isinstance(node.body[0], ast.Expr) 
                                    and isinstance(node.body[0].value, ast.Constant)):
                                    endpoint["description"] = node.body[0].value.value
                                
                                # Get parameters
                                for arg in node.args.args:
                                    if arg.arg not in ['self']:
                                        param = {
                                            "name": arg.arg,
                                            "type": "unknown",
                                            "required": True
                                        }
                                        
                                        # Try to get type annotation
                                        if arg.annotation:
                                            if isinstance(arg.annotation, ast.Name):
                                                param["type"] = arg.annotation.id
                                            elif isinstance(arg.annotation, ast.Attribute):
                                                param["type"] = f"{arg.annotation.value.id}.{arg.annotation.attr}"
                                        
                                        endpoint["parameters"].append(param)
                                
                                endpoints.append(endpoint)
                                break
                        
                        # Handle WebSocket endpoints
                        elif (isinstance(decorator, ast.Call) and 
                              hasattr(decorator.func, 'attr') and 
                              decorator.func.attr == 'websocket'):
                            
                            ws_endpoint = {
                                "path": "",
                                "function": node.name,
                                "description": "",
                                "type": "websocket"
                            }
                            
                            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                                ws_endpoint["path"] = decorator.args[0].value
                            
                            if (node.body and isinstance(node.body[0], ast.Expr) 
                                and isinstance(node.body[0].value, ast.Constant)):
                                ws_endpoint["description"] = node.body[0].value.value
                            
                            self.api_docs["websockets"].append(ws_endpoint)
        
        except Exception as e:
            print(f"Error extracting endpoints from {file_path}: {e}")
        
        return endpoints
    
    def extract_pydantic_models(self, file_path: str) -> List[Dict]:
        """Extract Pydantic models from models.py"""
        models = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from BaseModel
                    is_pydantic = False
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'BaseModel':
                            is_pydantic = True
                            break
                    
                    if is_pydantic:
                        model = {
                            "name": node.name,
                            "description": "",
                            "fields": []
                        }
                        
                        # Get docstring
                        if (node.body and isinstance(node.body[0], ast.Expr) 
                            and isinstance(node.body[0].value, ast.Constant)):
                            model["description"] = node.body[0].value.value
                        
                        # Get fields
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                field = {
                                    "name": item.target.id,
                                    "type": "unknown",
                                    "optional": False,
                                    "default": None
                                }
                                
                                # Get type annotation
                                if item.annotation:
                                    field["type"] = ast.unparse(item.annotation)
                                    
                                    # Check if Optional
                                    if "Optional" in field["type"]:
                                        field["optional"] = True
                                
                                # Get default value
                                if item.value:
                                    if isinstance(item.value, ast.Constant):
                                        field["default"] = item.value.value
                                    else:
                                        field["default"] = ast.unparse(item.value)
                                
                                model["fields"].append(field)
                        
                        models.append(model)
        
        except Exception as e:
            print(f"Error extracting models from {file_path}: {e}")
        
        return models
    
    def extract_class_methods(self, file_path: str, class_name: str = None) -> Dict:
        """Extract class methods and their documentation"""
        classes = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if class_name is None or node.name == class_name:
                        class_info = {
                            "name": node.name,
                            "description": "",
                            "methods": []
                        }
                        
                        # Get class docstring
                        if (node.body and isinstance(node.body[0], ast.Expr) 
                            and isinstance(node.body[0].value, ast.Constant)):
                            class_info["description"] = node.body[0].value.value
                        
                        # Get methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method = {
                                    "name": item.name,
                                    "description": "",
                                    "parameters": [],
                                    "returns": "unknown"
                                }
                                
                                # Get method docstring
                                if (item.body and isinstance(item.body[0], ast.Expr) 
                                    and isinstance(item.body[0].value, ast.Constant)):
                                    method["description"] = item.body[0].value.value
                                
                                # Get parameters
                                for arg in item.args.args:
                                    if arg.arg != 'self':
                                        param = {
                                            "name": arg.arg,
                                            "type": "unknown"
                                        }
                                        
                                        if arg.annotation:
                                            param["type"] = ast.unparse(arg.annotation)
                                        
                                        method["parameters"].append(param)
                                
                                # Get return type
                                if item.returns:
                                    method["returns"] = ast.unparse(item.returns)
                                
                                class_info["methods"].append(method)
                        
                        classes[node.name] = class_info
        
        except Exception as e:
            print(f"Error extracting classes from {file_path}: {e}")
        
        return classes
    
    def analyze_file_structure(self) -> Dict:
        """Analyze project file structure"""
        structure = {}
        
        def scan_directory(path: Path, level: int = 0) -> Dict:
            items = {}
            
            if level > 3:  # Limit depth
                return items
            
            try:
                for item in sorted(path.iterdir()):
                    if item.name.startswith('.') or item.name == '__pycache__':
                        continue
                    
                    if item.is_file():
                        items[item.name] = {
                            "type": "file",
                            "size": item.stat().st_size,
                            "extension": item.suffix,
                            "description": self.get_file_description(item)
                        }
                    elif item.is_dir():
                        items[item.name] = {
                            "type": "directory",
                            "contents": scan_directory(item, level + 1)
                        }
            except PermissionError:
                pass
            
            return items
        
        return scan_directory(self.project_root)
    
    def get_file_description(self, file_path: Path) -> str:
        """Get description for a file based on its content or name"""
        descriptions = {
            "main.py": "FastAPI backend server with REST API endpoints",
            "app.py": "Streamlit frontend application",
            "database.py": "Database connection and management",
            "models.py": "Pydantic data models and schemas",
            "config.py": "Application configuration settings",
            "area_picker.py": "Interactive area selection utility",
            "rtsp_handler.py": "RTSP stream handling and processing",
            "requirements.txt": "Python package dependencies",
            "README.md": "Project documentation and setup guide",
            ".env.example": "Environment variables template"
        }
        
        if file_path.name in descriptions:
            return descriptions[file_path.name]
        
        # Try to extract docstring from Python files
        if file_path.suffix == '.py':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                if (tree.body and isinstance(tree.body[0], ast.Expr) 
                    and isinstance(tree.body[0].value, ast.Constant)):
                    return tree.body[0].value.value.split('\n')[0]
            except:
                pass
        
        return f"{file_path.suffix[1:].upper()} file" if file_path.suffix else "File"
    
    def extract_configuration_options(self) -> Dict:
        """Extract configuration options from config files"""
        config = {}
        
        # Check config.py
        config_file = self.project_root / "backend" / "config.py"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                config[target.id] = {
                                    "value": ast.unparse(node.value),
                                    "type": type(node.value).__name__
                                }
            except Exception as e:
                print(f"Error extracting config: {e}")
        
        # Check .env.example
        env_file = self.project_root / ".env.example"
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config[key] = {
                                "value": value,
                                "type": "environment_variable",
                                "description": "Environment variable"
                            }
            except Exception as e:
                print(f"Error reading .env.example: {e}")
        
        return config
    
    def generate_api_documentation(self):
        """Generate API documentation"""
        print("Generating API documentation...")
        
        # Extract endpoints from main.py
        main_file = self.project_root / "backend" / "main.py"
        if main_file.exists():
            self.api_docs["endpoints"] = self.extract_fastapi_endpoints(str(main_file))
        
        # Extract models from models.py
        models_file = self.project_root / "backend" / "models.py"
        if models_file.exists():
            self.api_docs["models"] = self.extract_pydantic_models(str(models_file))
        
        # Save API documentation
        with open(self.docs_dir / "api_documentation.json", 'w', encoding='utf-8') as f:
            json.dump(self.api_docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ API documentation saved: {self.docs_dir}/api_documentation.json")
    
    def generate_system_documentation(self):
        """Generate system architecture documentation"""
        print("Generating system documentation...")
        
        # Analyze file structure
        self.system_docs["file_structure"] = self.analyze_file_structure()
        
        # Extract component information
        components = {}
        
        # Backend components
        backend_dir = self.project_root / "backend"
        if backend_dir.exists():
            for py_file in backend_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    classes = self.extract_class_methods(str(py_file))
                    if classes:
                        components[py_file.stem] = {
                            "file": str(py_file.relative_to(self.project_root)),
                            "classes": classes,
                            "type": "backend_component"
                        }
        
        # Utils components
        utils_dir = self.project_root / "utils"
        if utils_dir.exists():
            for py_file in utils_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    classes = self.extract_class_methods(str(py_file))
                    if classes:
                        components[py_file.stem] = {
                            "file": str(py_file.relative_to(self.project_root)),
                            "classes": classes,
                            "type": "utility_component"
                        }
        
        self.system_docs["components"] = components
        
        # Extract configuration
        self.system_docs["configuration"] = self.extract_configuration_options()
        
        # Add metadata
        self.system_docs["generated_at"] = datetime.now().isoformat()
        self.system_docs["project_root"] = str(self.project_root)
        
        # Save system documentation
        with open(self.docs_dir / "system_architecture.json", 'w', encoding='utf-8') as f:
            json.dump(self.system_docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ System documentation saved: {self.docs_dir}/system_architecture.json")
    
    def generate_usage_guide(self):
        """Generate usage guide and examples"""
        print("Generating usage guide...")
        
        # Installation steps
        self.usage_docs["installation"] = [
            {
                "step": 1,
                "title": "Clone Repository",
                "command": "git clone <repository-url>",
                "description": "Clone the warehouse stock counting repository"
            },
            {
                "step": 2,
                "title": "Install Dependencies",
                "command": "pip install -r requirements.txt",
                "description": "Install all required Python packages"
            },
            {
                "step": 3,
                "title": "Setup Environment",
                "command": "cp .env.example .env",
                "description": "Copy environment variables template and configure"
            },
            {
                "step": 4,
                "title": "Initialize Database",
                "command": "python scripts/setup_system.py",
                "description": "Setup database and create necessary tables"
            },
            {
                "step": 5,
                "title": "Start Services",
                "command": "python scripts/run_all.py",
                "description": "Start both backend and frontend services"
            }
        ]
        
        # Configuration examples
        self.usage_docs["configuration"] = [
            {
                "title": "Database Configuration",
                "description": "Configure database connection",
                "example": {
                    "SUPABASE_URL": "https://your-project.supabase.co",
                    "SUPABASE_ANON_KEY": "your-anon-key",
                    "DATABASE_URL": "postgresql://user:pass@localhost/warehouse"
                }
            },
            {
                "title": "RTSP Camera Setup",
                "description": "Configure RTSP camera streams",
                "example": {
                    "format": "rtsp://username:password@ip:port/stream",
                    "examples": [
                        "rtsp://admin:password123@192.168.1.100:554/stream1",
                        "rtsp://user:pass@camera.local:554/live"
                    ]
                }
            }
        ]
        
        # Usage examples
        self.usage_docs["examples"] = [
            {
                "title": "Upload Image for Area Definition",
                "method": "POST",
                "endpoint": "/upload-image",
                "description": "Upload warehouse image for area definition",
                "curl_example": "curl -X POST -F 'file=@warehouse.jpg' http://localhost:8000/upload-image"
            },
            {
                "title": "Define Monitoring Areas",
                "method": "POST",
                "endpoint": "/save-areas",
                "description": "Save defined pallet areas for monitoring",
                "json_example": {
                    "areas": [
                        [[100, 100], [200, 100], [200, 200], [100, 200]]
                    ],
                    "image_path": "/uploads/warehouse.jpg"
                }
            },
            {
                "title": "Start RTSP Monitoring",
                "method": "POST",
                "endpoint": "/start-rtsp-monitoring",
                "description": "Start real-time monitoring of RTSP stream",
                "json_example": {
                    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1"
                }
            }
        ]
        
        # Troubleshooting
        self.usage_docs["troubleshooting"] = [
            {
                "issue": "Port already in use",
                "solution": "Kill processes on ports 8000 and 8501",
                "commands": [
                    "lsof -ti:8000 | xargs kill -9",
                    "lsof -ti:8501 | xargs kill -9"
                ]
            },
            {
                "issue": "Database connection failed",
                "solution": "Check database credentials and network connectivity",
                "steps": [
                    "Verify environment variables in .env file",
                    "Test database connection manually",
                    "Check firewall and network settings"
                ]
            },
            {
                "issue": "OpenCV installation issues",
                "solution": "Install system dependencies for OpenCV",
                "commands": [
                    "sudo apt-get update",
                    "sudo apt-get install python3-opencv",
                    "pip install opencv-python"
                ]
            }
        ]
        
        # Add metadata
        self.usage_docs["generated_at"] = datetime.now().isoformat()
        self.usage_docs["version"] = "1.0.0"
        
        # Save usage documentation
        with open(self.docs_dir / "usage_guide.json", 'w', encoding='utf-8') as f:
            json.dump(self.usage_docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Usage guide saved: {self.docs_dir}/usage_guide.json")
    
    def generate_markdown_summary(self):
        """Generate a comprehensive markdown summary"""
        print("Generating markdown summary...")
        
        markdown_content = f"""# Warehouse Stock Counting System Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

The Warehouse Stock Counting System is a comprehensive solution for monitoring and counting stock in warehouse environments using computer vision and real-time video analysis.

## System Architecture

### Components

"""
        
        # Add component information
        for comp_name, comp_info in self.system_docs.get("components", {}).items():
            markdown_content += f"#### {comp_name.title()}\n"
            markdown_content += f"- **File**: `{comp_info['file']}`\n"
            markdown_content += f"- **Type**: {comp_info['type'].replace('_', ' ').title()}\n"
            
            if comp_info.get("classes"):
                markdown_content += "- **Classes**:\n"
                for class_name, class_info in comp_info["classes"].items():
                    markdown_content += f"  - `{class_name}`: {class_info.get('description', 'No description')}\n"
            
            markdown_content += "\n"
        
        # Add API endpoints
        markdown_content += "## API Endpoints\n\n"
        
        for endpoint in self.api_docs.get("endpoints", []):
            markdown_content += f"### {endpoint['method']} {endpoint['path']}\n"
            markdown_content += f"{endpoint.get('description', 'No description')}\n\n"
            
            if endpoint.get("parameters"):
                markdown_content += "**Parameters:**\n"
                for param in endpoint["parameters"]:
                    required = "Required" if param.get("required", True) else "Optional"
                    markdown_content += f"- `{param['name']}` ({param['type']}) - {required}\n"
                markdown_content += "\n"
        
        # Add WebSocket endpoints
        if self.api_docs.get("websockets"):
            markdown_content += "## WebSocket Endpoints\n\n"
            for ws in self.api_docs["websockets"]:
                markdown_content += f"### {ws['path']}\n"
                markdown_content += f"{ws.get('description', 'No description')}\n\n"
        
        # Add data models
        markdown_content += "## Data Models\n\n"
        
        for model in self.api_docs.get("models", []):
            markdown_content += f"### {model['name']}\n"
            markdown_content += f"{model.get('description', 'No description')}\n\n"
            
            if model.get("fields"):
                markdown_content += "**Fields:**\n"
                for field in model["fields"]:
                    optional = " (Optional)" if field.get("optional") else ""
                    default = f" = {field['default']}" if field.get("default") is not None else ""
                    markdown_content += f"- `{field['name']}`: {field['type']}{optional}{default}\n"
                markdown_content += "\n"
        
        # Add installation guide
        markdown_content += "## Installation\n\n"
        
        for step in self.usage_docs.get("installation", []):
            markdown_content += f"{step['step']}. **{step['title']}**\n"
            markdown_content += f"   ```bash\n   {step['command']}\n   ```\n"
            markdown_content += f"   {step['description']}\n\n"
        
        # Add configuration
        markdown_content += "## Configuration\n\n"
        
        for config in self.usage_docs.get("configuration", []):
            markdown_content += f"### {config['title']}\n"
            markdown_content += f"{config['description']}\n\n"
            markdown_content += "```json\n"
            markdown_content += json.dumps(config['example'], indent=2)
            markdown_content += "\n```\n\n"
        
        # Add troubleshooting
        markdown_content += "## Troubleshooting\n\n"
        
        for issue in self.usage_docs.get("troubleshooting", []):
            markdown_content += f"### {issue['issue']}\n"
            markdown_content += f"**Solution**: {issue['solution']}\n\n"
            
            if issue.get("commands"):
                markdown_content += "**Commands:**\n"
                for cmd in issue["commands"]:
                    markdown_content += f"```bash\n{cmd}\n```\n"
            
            if issue.get("steps"):
                markdown_content += "**Steps:**\n"
                for i, step in enumerate(issue["steps"], 1):
                    markdown_content += f"{i}. {step}\n"
            
            markdown_content += "\n"
        
        # Save markdown documentation
        with open(self.docs_dir / "DOCUMENTATION.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✅ Markdown summary saved: {self.docs_dir}/DOCUMENTATION.md")
    
    def generate_all(self):
        """Generate all documentation"""
        print("🚀 Starting documentation generation...")
        print("=" * 50)
        
        try:
            self.generate_api_documentation()
            self.generate_system_documentation()
            self.generate_usage_guide()
            self.generate_markdown_summary()
            
            print("\n" + "=" * 50)
            print("✅ Documentation generation completed successfully!")
            print(f"\nGenerated files in {self.docs_dir}:")
            print("- api_documentation.json")
            print("- system_architecture.json")
            print("- usage_guide.json")
            print("- DOCUMENTATION.md")
            
        except Exception as e:
            print(f"\n❌ Error during documentation generation: {e}")
            raise

def main():
    """Main function to run documentation generation"""
    generator = DocumentationGenerator()
    generator.generate_all()

if __name__ == "__main__":
    main()
