# Batmobile Tumbler — OpenGL

**Nombre:** Omar Alejandro Castellanos Gomez  
**Código:** 6014322

## Arquitectura del software

La aplicación se organiza en un archivo principal (`Main.cpp`), dividido en estructuras y funciones con responsabilidades específicas. La lógica de la escena se ejecuta en la CPU y el procesamiento gráfico se realiza en la GPU mediante shaders GLSL.

| Componente | Función |
| --- | --- |
| `main()` | Inicializa OpenGL, procesa las entradas y administra el bucle de renderizado. |
| `Viewer` | Carga y dibuja el modelo glTF con sus mallas y texturas mediante tinygltf. |
| `Platform` | Genera y dibuja la plataforma circular y el suelo. |
| `BezierPath` | Calcula el recorrido de 360°, la orientación del vehículo y la trayectoria visible. |
| `ControlPanel` | Gestiona la interfaz gráfica y los controles del mouse, manteniendo activo el teclado. |
| Vertex shader | Aplica las matrices de modelo, vista y proyección. |
| Fragment shader | Aplica las texturas y la iluminación Phong (ambiental, difusa y especular). |
