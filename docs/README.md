# Documentation

This folder contains documentation and diagrams for the GCP Cost Agent project.

## PlantUML Diagrams

### Architecture Diagrams

- **`architecture.puml`** - Overall system architecture showing the interaction between Frontend (React), FastAPI Backend, MCP Toolbox, and BigQuery
- **`detailed-flow.puml`** - Detailed sequence diagram showing the flow of a cost analysis request from user input to response
- **`deployment.puml`** - Cloud Run deployment process from local development to production

### Viewing the Diagrams

#### Option 1: Online Viewer
1. Copy the content of any `.puml` file
2. Paste it into [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
3. View the generated diagram

#### Option 2: Local Installation
1. Install PlantUML:
   - **Mac**: `brew install plantuml`
   - **Windows**: Download from [plantuml.com](https://plantuml.com/)
   - **Linux**: `sudo apt install plantuml`
2. Generate images: `plantuml *.puml`
3. View the generated PNG/SVG files

#### Option 3: VS Code Extension
1. Install "PlantUML" extension in VS Code
2. Open any `.puml` file
3. Use `Ctrl+Shift+P` → "PlantUML: Preview Current Diagram"

### Diagram Descriptions

#### architecture.puml
Shows the high-level system architecture including:
- Local development flow
- Cloud Run deployment flow
- Advanced analytics flow
- Error handling flow

#### detailed-flow.puml
Provides detailed sequence of operations for:
- Cost analysis requests
- Intent recognition with LLM
- MCP Toolbox interactions
- BigQuery queries
- Session management
- Error handling scenarios

#### deployment.puml
Illustrates the deployment process:
- Build and deploy workflow
- Cloud Build integration
- Container Registry
- Cloud Run configuration
- Runtime scaling
