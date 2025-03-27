# Dockerfile
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /home/alkain/Escritorio/Github_Repositorios/Microgrid

# Copia y instala las dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos
COPY . .

# Expone el puerto 5000 si se usa Flask
EXPOSE 5000

# Comando por defecto para ejecutar el servidor
CMD ["python", "inference.py"]
